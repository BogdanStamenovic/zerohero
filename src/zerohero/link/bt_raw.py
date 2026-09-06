"""RFCOMM sockets through libc when this Python was built without AF_BLUETOOTH.

uv's python-build-standalone interpreters (which install.py prefers) are
compiled without BlueZ headers, so `socket.AF_BLUETOOTH` does not exist even
though the kernel supports it. The kernel does not care how the fd was made:
create it with libc `socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM)`, do
bind/listen/accept/connect through libc with a hand-packed `sockaddr_rc`, and
only then hand the fd to `socket.socket(fileno=...)` for send/recv/close.
Anything address-related (getsockname, accept) on the wrapped object raises
"bad family" in CPython, which is why `RawListener` exists. Linux only.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import os
import select
import socket
import sys

AF_BLUETOOTH = 31
BTPROTO_RFCOMM = 3
_SOCK_STREAM = 1


class _SockaddrRC(ctypes.Structure):
    # struct sockaddr_rc { sa_family_t rc_family; bdaddr_t rc_bdaddr; uint8_t rc_channel; }
    _fields_ = [("rc_family", ctypes.c_ushort), ("rc_bdaddr", ctypes.c_ubyte * 6), ("rc_channel", ctypes.c_ubyte)]


def available() -> bool:
    return sys.platform.startswith("linux") and ctypes.util.find_library("c") is not None


_LIBC: ctypes.CDLL | None = None


def _libc() -> ctypes.CDLL:
    global _LIBC
    if _LIBC is None:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        libc.bind.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        libc.connect.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        libc.listen.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.accept.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        _LIBC = libc
    return _LIBC


def _bdaddr(addr: str) -> bytes:
    """'AA:BB:CC:DD:EE:FF' -> 6 bytes, least significant first, as bdaddr_t wants."""
    parts = addr.split(":")
    if len(parts) != 6:
        raise ValueError(f"bad bluetooth address {addr!r}")
    return bytes(int(p, 16) for p in reversed(parts))


def format_bdaddr(raw: bytes) -> str:
    return ":".join(f"{b:02X}" for b in reversed(raw[:6]))


def _sockaddr(addr: str, channel: int) -> _SockaddrRC:
    sa = _SockaddrRC()
    sa.rc_family = AF_BLUETOOTH
    sa.rc_bdaddr = (ctypes.c_ubyte * 6)(*_bdaddr(addr))
    sa.rc_channel = channel
    return sa


def _check(rc: int, what: str) -> int:
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")
    return rc


def _raw_socket() -> int:
    fd = _check(_libc().socket(AF_BLUETOOTH, _SOCK_STREAM, BTPROTO_RFCOMM), "socket(AF_BLUETOOTH)")
    # A fresh fd is blocking, but be explicit: connect() below relies on it.
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    return fd


def _wrap(fd: int) -> socket.socket:
    sock = socket.socket(fileno=fd)
    sock.setblocking(True)
    return sock


class RawListener:
    """Enough of the socket.socket listening API for Lead: settimeout, accept, close, fileno."""

    def __init__(self, fd: int, channel: int) -> None:
        self._fd = fd
        self.channel = channel
        self._timeout: float | None = None

    def fileno(self) -> int:
        return self._fd

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = timeout

    def accept(self) -> tuple[socket.socket, str]:
        if self._timeout is not None:
            ready, _, _ = select.select([self._fd], [], [], self._timeout)
            if not ready:
                raise TimeoutError("accept timed out")
        sa = _SockaddrRC()
        size = ctypes.c_uint(ctypes.sizeof(sa))
        fd = _check(_libc().accept(self._fd, ctypes.byref(sa), ctypes.byref(size)), "accept(rfcomm)")
        return _wrap(fd), f"{format_bdaddr(bytes(sa.rc_bdaddr))}/{sa.rc_channel}"

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def listen(channel: int, backlog: int = 8) -> RawListener:
    fd = _raw_socket()
    sa = _sockaddr("00:00:00:00:00:00", channel)  # BDADDR_ANY: whichever adapter
    try:
        _check(_libc().bind(fd, ctypes.byref(sa), ctypes.sizeof(sa)), "bind(rfcomm)")
        _check(_libc().listen(fd, backlog), "listen(rfcomm)")
    except OSError:
        os.close(fd)
        raise
    return RawListener(fd, channel)


def connect(addr: str, channel: int, timeout: float = 10) -> socket.socket:
    """Connect with a timeout: non-blocking connect, wait for writability, read SO_ERROR."""
    fd = _raw_socket()
    sa = _sockaddr(addr, channel)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        rc = _libc().connect(fd, ctypes.byref(sa), ctypes.sizeof(sa))
        if rc < 0 and ctypes.get_errno() != 115:  # EINPROGRESS is the expected answer
            _check(rc, f"connect({addr}/{channel})")
        _, writable, _ = select.select([], [fd], [], timeout)
        if not writable:
            raise TimeoutError(f"connect({addr}/{channel}): timed out after {timeout}s")
        sock = _wrap(fd)
        err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if err:
            sock.close()
            raise OSError(err, f"connect({addr}/{channel}): {os.strerror(err)}")
        return sock
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
