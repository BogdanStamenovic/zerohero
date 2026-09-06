"""Stream socket wrapper + TCP/RFCOMM listen and connect. See ARCHITECTURE.md "link"."""

from __future__ import annotations

import socket
import struct
import threading

from zerohero.link.protocol import Message, decode, encode

# SO_SNDTIMEO bounds only the send path (Linux keeps send/recv timeouts
# separate at the socket-option level, unlike socket.settimeout() which sets
# both). This lets Lead.send_chord/send_strum block briefly on a wedged
# follower instead of hanging the caller, while recv() stays a plain
# indefinite blocking read for the dedicated reader threads.
_SEND_TIMEOUT = struct.pack("ll", 2, 0)  # 2s, 0us


class BluetoothUnavailable(RuntimeError):
    def __init__(self, msg: str = "") -> None:
        super().__init__(
            msg
            or "Bluetooth RFCOMM sockets are not available on this Python build "
            "(no socket.AF_BLUETOOTH). This needs Linux with BlueZ; on other "
            "platforms, or Pythons built without bluetooth headers, use --transport tcp."
        )


class Connection:
    """A connected stream socket speaking newline-delimited JSON.

    One reader (whoever calls recv()) and any number of writers; send() is
    locked so followers/lead threads can share a Connection safely.
    """

    def __init__(self, sock: socket.socket, peer: str) -> None:
        self._sock = sock
        self.peer = peer
        self._send_lock = threading.Lock()
        self._buf = b""
        self._closed = False
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, _SEND_TIMEOUT)
        except OSError:
            pass  # not supported on this platform/socket family; sends just block longer

    def send(self, msg: Message) -> bool:
        """Returns False (and marks the connection closed) on error instead of raising,
        so a dead follower can never throw into the lead's main loop."""
        data = encode(msg)
        with self._send_lock:
            if self._closed:
                return False
            try:
                self._sock.sendall(data)
                return True
            except OSError:
                self._closed = True
                return False

    def recv(self) -> Message | None:
        """Blocking read of one message. Returns None on EOF or a closed/broken socket."""
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(4096)
            except OSError:
                return None
            if not chunk:
                return None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return decode(line)

    def close(self) -> None:
        with self._send_lock:
            self._closed = True
        # shutdown() (unlike close()) reliably unblocks a peer thread parked in
        # recv() on this same socket on Linux - close() alone can leave that
        # thread stuck until the kernel notices, which is not bounded in time.
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def closed(self) -> bool:
        return self._closed


def listen_tcp(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(8)
    return sock


def connect_tcp(host: str, port: int, timeout: float = 5) -> Connection:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)  # blocking recv() after the handshake connects
    return Connection(sock, f"{host}:{port}")


def listen_bt(channel: int) -> socket.socket:
    if not hasattr(socket, "AF_BLUETOOTH"):
        raise BluetoothUnavailable()
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.bind(("", channel))
    sock.listen(8)
    return sock


def connect_bt(addr: str, channel: int, timeout: float = 10) -> Connection:
    if not hasattr(socket, "AF_BLUETOOTH"):
        raise BluetoothUnavailable()
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.settimeout(timeout)
    sock.connect((addr, channel))
    sock.settimeout(None)
    return Connection(sock, f"{addr}/{channel}")


def parse_target(s: str, default_port: int, default_channel: int) -> tuple:
    """Parse a --follow target string.

    Forms: "host" -> ("tcp", host, default_port)
           "host:port" -> ("tcp", host, port)
           "bt:AA:BB:CC:DD:EE:FF" -> ("bt", addr, default_channel)
           "bt:AA:BB:CC:DD:EE:FF/5" -> ("bt", addr, 5)
    """
    if s.startswith("bt:"):
        rest = s[len("bt:") :]
        addr, sep, chan = rest.partition("/")
        return ("bt", addr, int(chan) if sep else default_channel)
    host, sep, port = s.rpartition(":")
    if sep and port.isdigit():
        return ("tcp", host, int(port))
    return ("tcp", s, default_port)
