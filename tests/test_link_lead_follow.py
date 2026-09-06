import socket
import threading
import time

from zerohero.config import LinkConfig
from zerohero.link.follow import Follower
from zerohero.link.lead import Lead


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until(predicate, timeout=3.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_lead(name="lead") -> Lead:
    cfg = LinkConfig(port=0, discovery_port=_free_udp_port(), name=name)
    lead = Lead(cfg, "tcp")
    lead.start()
    return lead


def _make_follower(lead: Lead, name="follow") -> Follower:
    cfg = LinkConfig(port=lead.port, discovery_port=_free_udp_port())
    follower = Follower(cfg, f"127.0.0.1:{lead.port}", name)
    return follower


def test_chord_and_strum_delivered_in_order():
    lead = _make_lead()
    follower = _make_follower(lead)
    received: list[tuple] = []
    follower.on_chord = lambda index, symbol, t_local: received.append(("chord", index, symbol))
    follower.on_strum = lambda direction, intensity, t_local: received.append(("strum", direction, intensity))
    follower.start()
    try:
        assert _wait_until(lambda: follower.connected)
        assert _wait_until(lambda: lead.followers == 1)

        lead.send_chord(0, "Am")
        lead.send_strum("down", 0.7)
        lead.send_chord(1, "F")

        assert _wait_until(lambda: len(received) == 3)
        assert received == [
            ("chord", 0, "Am"),
            ("strum", "down", 0.7),
            ("chord", 1, "F"),
        ]
    finally:
        follower.stop()
        lead.stop()


def test_offset_and_rtt_on_loopback():
    lead = _make_lead()
    follower = _make_follower(lead)
    follower.start()
    try:
        assert _wait_until(lambda: follower.offset is not None, timeout=3.0)
        assert abs(follower.offset) < 0.02  # within 20ms on loopback
        assert follower.rtt_ms is not None
        assert follower.rtt_ms < 50  # loopback should be well under this
    finally:
        follower.stop()
        lead.stop()


def test_tempo_estimate_from_four_strums_half_second_apart():
    lead = _make_lead()
    follower = _make_follower(lead)
    follower.start()
    try:
        assert _wait_until(lambda: lead.followers == 1)
        assert follower.tempo is None  # nothing sent yet

        for _ in range(4):
            lead.send_strum("down", 0.5)
            time.sleep(0.5)

        tempo = follower.tempo
        assert tempo is not None
        assert 110 < tempo < 130  # ~120 BPM from 0.5s inter-onset interval
    finally:
        follower.stop()
        lead.stop()


def test_tempo_none_with_few_or_stale_strums():
    lead = _make_lead()
    follower = _make_follower(lead)
    follower.start()
    try:
        assert _wait_until(lambda: lead.followers == 1)
        lead.send_strum("down", 0.5)
        time.sleep(0.05)
        lead.send_strum("down", 0.5)
        assert _wait_until(lambda: len(follower._strum_times) == 2, timeout=1)
        assert follower.tempo is None  # fewer than 3 strums
    finally:
        follower.stop()
        lead.stop()


def test_bye_from_lead_triggers_on_disconnect():
    lead = _make_lead()
    follower = _make_follower(lead)
    events = []
    follower.on_connect = lambda: events.append("connect")
    follower.on_disconnect = lambda: events.append("disconnect")
    follower.start()
    try:
        assert _wait_until(lambda: follower.connected)
        lead.stop()  # sends Bye to followers, then closes
        assert _wait_until(lambda: not follower.connected)
        assert "disconnect" in events
    finally:
        follower.stop()


def test_follower_reconnects_to_new_lead_on_same_port():
    lead1 = _make_lead()
    port = lead1.port
    discovery_port = lead1.cfg.discovery_port

    cfg = LinkConfig(port=port, discovery_port=discovery_port)
    follower = Follower(cfg, f"127.0.0.1:{port}", "follow")
    connects = []
    follower.on_connect = lambda: connects.append(1)
    follower.start()
    try:
        assert _wait_until(lambda: follower.connected)
        lead1.stop()
        assert _wait_until(lambda: not follower.connected)

        # a fresh Lead bound to the exact same TCP port
        lead2 = Lead(LinkConfig(port=port, discovery_port=discovery_port, name="lead2"), "tcp")
        try:
            lead2.start()
        except OSError:
            # port not free yet right after close(); give the OS a moment and retry once
            time.sleep(0.3)
            lead2 = Lead(LinkConfig(port=port, discovery_port=discovery_port, name="lead2"), "tcp")
            lead2.start()

        assert _wait_until(lambda: follower.connected, timeout=4.0)
        assert len(connects) == 2
        lead2.stop()
    finally:
        follower.stop()


def test_two_followers_receive_broadcast():
    lead = _make_lead()
    f1 = _make_follower(lead, "f1")
    f2 = _make_follower(lead, "f2")
    got1, got2 = [], []
    f1.on_chord = lambda index, symbol, t_local: got1.append(symbol)
    f2.on_chord = lambda index, symbol, t_local: got2.append(symbol)
    f1.start()
    f2.start()
    try:
        assert _wait_until(lambda: lead.followers == 2)
        lead.send_chord(0, "G")
        assert _wait_until(lambda: got1 == ["G"] and got2 == ["G"])
    finally:
        f1.stop()
        f2.stop()
        lead.stop()


def test_stop_immediately_after_connect_never_hangs():
    """Regression: a connection whose TCP handshake completes right as stop()
    runs can be accept()-ed *after* stop() already snapshotted and Bye'd the
    connection list. If that straggler got registered anyway it would never
    receive a Bye or a close, hanging both the lead's reader thread and the
    follower's read loop forever. Hammer the connect/stop boundary to catch
    that race; runs the check in a thread with its own timeout so a
    regression fails fast instead of hanging the whole suite.
    """

    def hammer():
        for _ in range(25):
            lead = _make_lead()
            follower = _make_follower(lead)
            follower.start()
            assert _wait_until(lambda f=follower: f.connected, timeout=2.0)
            lead.stop()  # racing the lead's accept thread by design
            assert _wait_until(lambda f=follower: not f.connected, timeout=2.0)
            follower.stop()

    result: list[BaseException] = []

    def run():
        try:
            hammer()
        except BaseException as e:  # noqa: BLE001 - reported to the test thread below
            result.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=4.5)
    assert not t.is_alive(), "hammer loop hung - the accept/stop race regressed"
    if result:
        raise result[0]
