"""Lead side of link: accept followers, broadcast chord/strum. See ARCHITECTURE.md "link"."""

from __future__ import annotations

import logging
import socket
import threading
import time

from zerohero.config import LinkConfig
from zerohero.link.discovery import Beacon
from zerohero.link.protocol import Bye, ChordMsg, Hello, Message, Ping, Pong, ProtocolError, StrumMsg
from zerohero.link.transport import Connection, listen_bt, listen_tcp

log = logging.getLogger(__name__)


class Lead:
    """Owns the listening socket and one reader thread per connected follower.

    Threading model: the accept thread owns the listen socket and spawns a
    reader thread per connection; each reader thread owns recv() on its own
    connection and only that one. send_chord/send_strum run on the caller's
    thread (the main loop) and write to every connection under `_lock` -
    Connection.send() is itself locked and time-bounded (SO_SNDTIMEO), so a
    wedged follower costs the caller at most ~2s instead of hanging forever,
    and never raises into the main loop.
    """

    def __init__(self, cfg: LinkConfig, transport: str = "tcp") -> None:
        self.cfg = cfg
        self.transport = transport
        self._listen_sock: socket.socket | None = None
        self._beacon: Beacon | None = None
        self._accept_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._conns: dict[Connection, str] = {}  # conn -> follower name
        self._stop = threading.Event()

    def start(self) -> None:
        if self.transport == "tcp":
            self._listen_sock = listen_tcp(self.cfg.port)
            self._beacon = Beacon(self.cfg.port, self.cfg.name, self.cfg.discovery_port)
            self._beacon.start()
        elif self.transport == "bt":
            self._listen_sock = listen_bt(self.cfg.bt_channel)
        else:
            raise ValueError(f"unknown transport: {self.transport}")
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._listen_sock is not None
        # A short accept() timeout, not a reliance on stop() closing the
        # socket to unblock us: closing a listening socket from another
        # thread while this one is parked in accept() is not guaranteed to
        # wake it promptly on Linux, which made stop() take as long as
        # whatever join() timeout was chosen instead of returning quickly.
        self._listen_sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                sock, addr = self._listen_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return  # listen socket closed by stop()
            conn = Connection(sock, str(addr))
            with self._lock:
                # accept() can still hand back a connection whose handshake
                # completed just before stop() closed the listen socket - it
                # was already in the kernel backlog, not something close()
                # drops. Registering it here under the same lock stop() uses
                # to snapshot-and-clear makes the two mutually exclusive: if
                # stop() already ran, is_set() is True by the time we get the
                # lock, so we drop the connection unregistered instead of
                # leaving it to hang forever with no Bye and no reader.
                if self._stop.is_set():
                    sock.close()
                    return
                self._conns[conn] = ""
            threading.Thread(target=self._reader, args=(conn,), daemon=True).start()

    def _reader(self, conn: Connection) -> None:
        try:
            first = conn.recv()
            if isinstance(first, Hello):
                with self._lock:
                    self._conns[conn] = first.name
                log.info("follower connected: %s (%s)", first.name, conn.peer)
            elif first is None:
                self._drop(conn)
                return
            # else: not a Hello - keep going, tolerate a misbehaving peer

            while True:
                try:
                    msg = conn.recv()
                except ProtocolError:
                    continue  # skip one garbled line, connection stays up
                if msg is None or isinstance(msg, Bye):
                    break
                if isinstance(msg, Ping):
                    conn.send(Pong(t=msg.t, rt=time.monotonic()))
                # ChordMsg/StrumMsg/Hello from a follower isn't part of the
                # protocol (those flow lead -> follower); ignore silently.
        except Exception:
            log.exception("link reader error for %s", conn.peer)
        finally:
            self._drop(conn)

    def _drop(self, conn: Connection) -> None:
        with self._lock:
            self._conns.pop(conn, None)
        conn.close()

    def send_chord(self, index: int, symbol: str) -> None:
        self._broadcast(ChordMsg(index=index, symbol=symbol, t=time.monotonic()))

    def send_strum(self, direction: str, intensity: float) -> None:
        self._broadcast(StrumMsg(direction=direction, intensity=intensity, t=time.monotonic()))

    def _broadcast(self, msg: Message) -> None:
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            conn.send(msg)  # never raises; failures self-close the connection

    @property
    def followers(self) -> int:
        with self._lock:
            return len(self._conns)

    @property
    def port(self) -> int:
        """Actual bound TCP port (useful with cfg.port=0, e.g. in tests)."""
        assert self._listen_sock is not None, "call start() first"
        return self._listen_sock.getsockname()[1]

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            conns = list(self._conns)
            self._conns.clear()
        for conn in conns:
            conn.send(Bye())
            conn.close()
        if self._beacon is not None:
            self._beacon.stop()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
