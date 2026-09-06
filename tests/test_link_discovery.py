import socket

from zerohero.link.discovery import Beacon, discover


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_discover_finds_beacon_on_loopback():
    discovery_port = _free_udp_port()
    beacon = Beacon(port=47999, name="test-lead", discovery_port=discovery_port)
    beacon.start()
    try:
        found = discover(discovery_port, timeout=2.5)
    finally:
        beacon.stop()
    assert len(found) >= 1
    hosts = {f[0] for f in found}
    ports = {f[1] for f in found}
    names = {f[2] for f in found}
    assert "127.0.0.1" in hosts
    assert 47999 in ports
    assert "test-lead" in names


def test_discover_times_out_with_no_beacon():
    discovery_port = _free_udp_port()
    found = discover(discovery_port, timeout=0.3)
    assert found == []


def test_discover_deduplicates_by_host():
    discovery_port = _free_udp_port()
    beacon = Beacon(port=48111, name="dup-lead", discovery_port=discovery_port)
    beacon.start()
    try:
        found = discover(discovery_port, timeout=2.2)
    finally:
        beacon.stop()
    host_counts: dict[str, int] = {}
    for host, _port, _name in found:
        host_counts[host] = host_counts.get(host, 0) + 1
    assert all(count == 1 for count in host_counts.values())
