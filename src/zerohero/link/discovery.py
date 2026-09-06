"""UDP broadcast beacon + discovery. See ARCHITECTURE.md "link"."""

from __future__ import annotations

import json
import socket
import threading
import time


class Beacon:
    """Broadcasts the lead's presence once a second until stop().

    Sends to both 255.255.255.255 and 127.0.0.1: real broadcast for LAN
    discovery, plus loopback so a lead and follower on the same machine (e.g.
    tests, or a sandboxed dev box that blocks broadcast) can find each other
    without a network.
    """

    def __init__(self, port: int, name: str, discovery_port: int) -> None:
        self.port = port
        self.name = name
        self.discovery_port = discovery_port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        payload = json.dumps({"zerohero": 1, "port": self.port, "name": self.name}).encode("utf-8")
        try:
            while not self._stop.is_set():
                for dest in ("255.255.255.255", "127.0.0.1"):
                    try:
                        sock.sendto(payload, (dest, self.discovery_port))
                    except OSError:
                        pass  # no network / no route; keep trying next tick
                self._stop.wait(1.0)
        finally:
            sock.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def discover(discovery_port: int, timeout: float) -> list[tuple[str, int, str]]:
    """Listen for beacons for `timeout` seconds, deduplicated by host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("0.0.0.0", discovery_port))
    sock.settimeout(0.2)
    found: dict[str, tuple[str, int, str]] = {}
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                data, (host, _port) = sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                d = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(d, dict) or d.get("zerohero") != 1:
                continue
            found[host] = (host, int(d["port"]), str(d.get("name", "")))
    finally:
        sock.close()
    return list(found.values())
