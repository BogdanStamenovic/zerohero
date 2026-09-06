from zerohero.link.protocol import (
    Bye,
    ChordMsg,
    Hello,
    Ping,
    Pong,
    ProtocolError,
    StrumMsg,
    decode,
    encode,
)

MESSAGES = [
    Hello(role="lead", name="guitar-laptop", version=1),
    Hello(role="follow", name="piano-tablet"),
    ChordMsg(index=3, symbol="F#m7", t=12.5),
    StrumMsg(direction="down", intensity=0.8, t=12.6),
    StrumMsg(direction="up", intensity=0.1, t=12.7),
    Ping(t=1.0),
    Pong(t=1.0, rt=1.001),
    Bye(),
]


def test_roundtrip_all_message_types():
    for msg in MESSAGES:
        encoded = encode(msg)
        assert encoded.endswith(b"\n")
        assert encoded.count(b"\n") == 1
        decoded = decode(encoded.rstrip(b"\n"))
        assert decoded == msg


def test_encode_is_one_json_object_per_line():
    encoded = encode(ChordMsg(index=0, symbol="Am", t=0.0))
    line = encoded.rstrip(b"\n").decode()
    assert line.count("\n") == 0
    assert '"type":"chord"' in line


def test_decode_garbage_raises_protocol_error():
    for bad in (b"not json", b"{}", b'{"type": "unknown_type"}', b'{"type": "hello"}', b""):
        try:
            decode(bad)
        except ProtocolError:
            continue
        raise AssertionError(f"expected ProtocolError for {bad!r}")


def test_decode_extra_or_missing_fields_raise():
    try:
        decode(b'{"type":"chord","index":1}')  # missing symbol/t
    except ProtocolError:
        pass
    else:
        raise AssertionError("expected ProtocolError for missing fields")
