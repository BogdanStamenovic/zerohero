import socket
import threading
import time

import pytest

from zerohero.link.protocol import ChordMsg, Hello, Ping
from zerohero.link.transport import (
    BluetoothUnavailable,
    Connection,
    connect_tcp,
    listen_tcp,
    parse_target,
)


def _connection_pair() -> tuple[Connection, Connection]:
    a, b = socket.socketpair()
    return Connection(a, "a"), Connection(b, "b")


def test_framing_message_split_across_two_sends():
    a, b = _connection_pair()
    try:
        line = b'{"type":"ping","t":1.5}\n'
        a._sock.sendall(line[:10])
        a._sock.sendall(line[10:])
        msg = b.recv()
        assert msg == Ping(t=1.5)
    finally:
        a.close()
        b.close()


def test_framing_two_messages_in_one_send():
    a, b = _connection_pair()
    try:
        payload = b'{"type":"ping","t":1.0}\n{"type":"ping","t":2.0}\n'
        a._sock.sendall(payload)
        assert b.recv() == Ping(t=1.0)
        assert b.recv() == Ping(t=2.0)
    finally:
        a.close()
        b.close()


def test_send_recv_round_trip_via_connection_api():
    a, b = _connection_pair()
    try:
        assert a.send(Hello(role="lead", name="x")) is True
        assert b.recv() == Hello(role="lead", name="x")
        assert b.send(ChordMsg(index=1, symbol="Am", t=3.0)) is True
        assert a.recv() == ChordMsg(index=1, symbol="Am", t=3.0)
    finally:
        a.close()
        b.close()


def test_recv_returns_none_on_close():
    a, b = _connection_pair()
    a.close()
    assert b.recv() is None
    b.close()


def test_send_after_close_returns_false():
    a, b = _connection_pair()
    a.close()
    assert a.send(Ping(t=0.0)) is False
    b.close()


def test_tcp_listen_connect_roundtrip():
    srv = listen_tcp(0)
    port = srv.getsockname()[1]
    accepted: list[Connection] = []

    def accept_one():
        sock, addr = srv.accept()
        accepted.append(Connection(sock, str(addr)))

    t = threading.Thread(target=accept_one, daemon=True)
    t.start()
    client = connect_tcp("127.0.0.1", port, timeout=2)
    t.join(timeout=2)
    assert accepted
    server_conn = accepted[0]
    try:
        assert client.send(Hello(role="follow", name="x")) is True
        assert server_conn.recv() == Hello(role="follow", name="x")
    finally:
        client.close()
        server_conn.close()
        srv.close()


def test_tcp_nodelay_is_set():
    srv = listen_tcp(0)
    port = srv.getsockname()[1]
    t = threading.Thread(target=srv.accept, daemon=True)
    t.start()
    client = connect_tcp("127.0.0.1", port, timeout=2)
    time.sleep(0.05)
    nodelay = client._sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
    assert nodelay != 0
    client.close()
    srv.close()


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("myhost", ("tcp", "myhost", 47475)),
        ("myhost:9000", ("tcp", "myhost", 9000)),
        ("192.168.1.5", ("tcp", "192.168.1.5", 47475)),
        ("192.168.1.5:9000", ("tcp", "192.168.1.5", 9000)),
        ("bt:AA:BB:CC:DD:EE:FF", ("bt", "AA:BB:CC:DD:EE:FF", 3)),
        ("bt:AA:BB:CC:DD:EE:FF/5", ("bt", "AA:BB:CC:DD:EE:FF", 5)),
    ],
)
def test_parse_target(s, expected):
    assert parse_target(s, default_port=47475, default_channel=3) == expected


def test_bluetooth_unavailable_when_no_af_bluetooth():
    if hasattr(socket, "AF_BLUETOOTH"):
        pytest.skip("this Python build has AF_BLUETOOTH; BluetoothUnavailable path not exercised")
    from zerohero.link.transport import connect_bt, listen_bt

    with pytest.raises(BluetoothUnavailable):
        listen_bt(3)
    with pytest.raises(BluetoothUnavailable):
        connect_bt("AA:BB:CC:DD:EE:FF", 3)
