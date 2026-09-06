"""Follower side of link: connect, clock offset, tempo estimate. See ARCHITECTURE.md "link"."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from statistics import median

from zerohero.config import LinkConfig
from zerohero.link.discovery import discover
from zerohero.link.protocol import Bye, ChordMsg, Hello, Ping, Pong, ProtocolError, StrumMsg
from zerohero.link.transport import Connection, connect_bt, connect_tcp, parse_target

log = logging.getLogger(__name__)

_RECONNECT_DELAY = 2.0
_STEADY_PING_INTERVAL = 2.0
_INITIAL_PING_COUNT = 5
_INITIAL_PING_GAP = 0.05
_OFFSET_SAMPLES = 9
_TEMPO_SAMPLES = 8
_TEMPO_STALE_S = 3.0


class Follower:
    """Connects to a Lead, keeps a clock-offset estimate, and estimates tempo from strums.

    Threading model: one worker thread owns the connect/reconnect loop and the
    blocking read loop; it spawns one ping thread per connection attempt.
    Callback attributes (on_chord etc.) are invoked from the worker thread, so
    they must not block for long or they'll delay the next recv().
    """

    def __init__(self, cfg: LinkConfig, target: str, name: str) -> None:
        self.cfg = cfg
        self.target = target
        self.name = name

        self.on_chord: Callable[[int, str, float], None] = lambda index, symbol, t_local: None
        self.on_strum: Callable[[str, float, float], None] = lambda direction, intensity, t_local: None
        self.on_connect: Callable[[], None] = lambda: None
        self.on_disconnect: Callable[[], None] = lambda: None

        self._resolved: tuple | None = None
        self._conn: Connection | None = None
        self._connected = False
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        self._offset_lock = threading.Lock()
        self._offsets: deque[float] = deque(maxlen=_OFFSET_SAMPLES)
        self._offset: float | None = None
        self._rtt_ms: float | None = None

        self._strum_times: deque[float] = deque(maxlen=_TEMPO_SAMPLES)

    def start(self) -> None:
        self._resolved = self._resolve(self.target)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _resolve(self, target: str) -> tuple:
        if target == "auto":
            found = discover(self.cfg.discovery_port, self.cfg.discover_timeout)
            if not found:
                raise RuntimeError("no lead found via discovery (auto)")
            host, port, _name = found[0]
            return ("tcp", host, port)
        return parse_target(target, self.cfg.port, self.cfg.bt_channel)

    def _run(self) -> None:
        assert self._resolved is not None
        kind, addr, port_or_chan = self._resolved
        while not self._stop.is_set():
            try:
                conn = connect_tcp(addr, port_or_chan) if kind == "tcp" else connect_bt(addr, port_or_chan)
            except OSError as e:
                log.debug("connect failed: %s", e)
                if self._stop.wait(_RECONNECT_DELAY):
                    return
                continue

            self._conn = conn
            with self._offset_lock:
                self._offsets.clear()
                self._offset = None
                self._rtt_ms = None
            self._strum_times.clear()

            if not conn.send(Hello(role="follow", name=self.name)):
                conn.close()
                if self._stop.wait(_RECONNECT_DELAY):
                    return
                continue

            self._connected = True
            ping_thread = threading.Thread(target=self._ping_loop, args=(conn,), daemon=True)
            ping_thread.start()
            self.on_connect()

            self._read_loop(conn)

            self._connected = False
            conn.close()
            ping_thread.join(timeout=1)
            self.on_disconnect()

            if self._stop.is_set():
                return
            self._stop.wait(_RECONNECT_DELAY)

    def _ping_loop(self, conn: Connection) -> None:
        for _ in range(_INITIAL_PING_COUNT):
            if self._stop.is_set() or conn.closed:
                return
            if not conn.send(Ping(t=time.monotonic())):
                return
            time.sleep(_INITIAL_PING_GAP)
        while not self._stop.is_set() and not conn.closed:
            if not conn.send(Ping(t=time.monotonic())):
                return
            if self._stop.wait(_STEADY_PING_INTERVAL):
                return

    def _read_loop(self, conn: Connection) -> None:
        while True:
            try:
                msg = conn.recv()
            except ProtocolError:
                continue  # skip one garbled line, connection stays up
            if msg is None or isinstance(msg, Bye):
                return
            if isinstance(msg, Pong):
                self._handle_pong(msg)
            elif isinstance(msg, ChordMsg):
                self.on_chord(msg.index, msg.symbol, self._to_local(msg.t))
            elif isinstance(msg, StrumMsg):
                t_local = self._to_local(msg.t)
                self._strum_times.append(t_local)
                self.on_strum(msg.direction, msg.intensity, t_local)
            # Hello/Ping arriving here would be a lead bug; ignore rather than crash.

    def _handle_pong(self, pong: Pong) -> None:
        t_recv = time.monotonic()
        # Clock offset sign convention: offset := lead_clock - follower_clock,
        # estimated at the round-trip midpoint (t_send + t_recv) / 2 assuming
        # symmetric latency. pong.t is the follower's own send time (echoed
        # back unchanged), pong.rt is the lead's clock when it answered.
        # To map a lead timestamp into follower-local time: t_local = t_lead - offset.
        offset_sample = pong.rt - (pong.t + t_recv) / 2
        with self._offset_lock:
            self._offsets.append(offset_sample)
            self._offset = median(self._offsets)
            self._rtt_ms = (t_recv - pong.t) * 1000.0

    def _to_local(self, t_lead: float) -> float:
        offset = self.offset
        return t_lead - offset if offset is not None else t_lead

    @property
    def offset(self) -> float | None:
        with self._offset_lock:
            return self._offset

    @property
    def rtt_ms(self) -> float | None:
        with self._offset_lock:
            return self._rtt_ms

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tempo(self) -> float | None:
        times = list(self._strum_times)
        if len(times) < 3:
            return None
        if time.monotonic() - times[-1] > _TEMPO_STALE_S:
            return None
        intervals = [b - a for a, b in zip(times, times[1:], strict=False)]
        return 60.0 / median(intervals)

    def stop(self) -> None:
        self._stop.set()
        conn = self._conn
        if conn is not None:
            conn.send(Bye())
            conn.close()
        if self._worker is not None:
            self._worker.join(timeout=3)
