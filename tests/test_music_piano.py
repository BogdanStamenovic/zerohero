import pytest

from zerohero.config import MusicConfig
from zerohero.events import Beat
from zerohero.music.piano import PianoVoicer, accompaniment, phrase
from zerohero.music.theory import chord_scale, midi, parse_chord

CFG = MusicConfig()


def _beat(hand="left", direction="left", intensity=0.6, closed=False) -> Beat:
    return Beat(t=0.0, hand=hand, direction=direction, intensity=intensity, closed=closed)


def test_voicing_stays_in_register_range() -> None:
    voicer = PianoVoicer()
    low = voicer.voicing(parse_chord("C"), "low", 0.0)
    high = voicer.voicing(parse_chord("C"), "high", 0.0)
    assert all(36 <= n <= 55 for n in low)
    assert all(55 <= n <= 79 for n in high)


def test_voicing_contains_only_chord_tones_at_zero_density() -> None:
    chord = parse_chord("Am7")
    voicer = PianoVoicer()
    notes = voicer.voicing(chord, "low", 0.0)
    pcs = set(chord.pitch_classes())
    assert notes  # non-empty
    for n in notes:
        assert n % 12 in pcs


def test_density_adds_extensions() -> None:
    chord = parse_chord("C")
    voicer = PianoVoicer()
    plain = set(n % 12 for n in voicer.voicing(chord, "low", 0.0))
    with_seventh = set(n % 12 for n in PianoVoicer().voicing(chord, "low", 0.5))
    with_ninth = set(n % 12 for n in PianoVoicer().voicing(chord, "low", 0.7))
    assert plain == {0, 4, 7}
    assert with_seventh >= plain
    assert len(with_seventh) > len(plain)
    assert len(with_ninth) >= len(with_seventh)


def test_voice_leading_minimises_movement_between_related_chords() -> None:
    voicer = PianoVoicer()
    c_voicing = voicer.voicing(parse_chord("C"), "low", 0.0)
    f_voicing = voicer.voicing(parse_chord("F"), "low", 0.0)
    movement = sum(abs(a - b) for a, b in zip(sorted(c_voicing), sorted(f_voicing), strict=True))
    # C and F share a tone (C) and are a fifth apart; a good voicing keeps
    # total movement well under a full root-position jump (12+ semitones).
    assert movement <= 6


def test_registers_kept_separately_in_memory() -> None:
    voicer = PianoVoicer()
    voicer.voicing(parse_chord("C"), "low", 0.0)
    voicer.voicing(parse_chord("G"), "high", 0.0)
    assert voicer._memory["low"] and all(n <= 55 for n in voicer._memory["low"])
    assert voicer._memory["high"] and all(n >= 55 for n in voicer._memory["high"])


@pytest.mark.parametrize("hand,expected_low", [("left", True), ("right", False)])
def test_register_follows_hand(hand: str, expected_low: bool) -> None:
    chord = parse_chord("C")
    voicer = PianoVoicer()
    events = phrase(chord, _beat(hand=hand, direction="left"), vibe=0.2, tempo=None, voicer=voicer, cfg=CFG)
    assert events
    is_low = all(n < 55 for n in (e.note for e in events))
    assert is_low == expected_low


def test_left_direction_is_block_chord_at_offset_zero_with_humanisation() -> None:
    chord = parse_chord("C")
    events = phrase(chord, _beat(direction="left"), vibe=0.2, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    assert len(events) >= 3
    for e in events:
        assert 0 <= e.offset <= 0.008


def test_right_direction_is_ascending_arpeggio() -> None:
    chord = parse_chord("C")
    events = phrase(chord, _beat(direction="right"), vibe=0.2, tempo=120, voicer=PianoVoicer(), cfg=CFG)
    notes = [e.note for e in events]
    offsets = [e.offset for e in events]
    assert notes == sorted(notes)
    assert offsets == sorted(offsets)
    assert offsets[1] - offsets[0] == pytest.approx(60 / 120 / 4)


def test_right_direction_spacing_without_tempo_uses_intensity() -> None:
    chord = parse_chord("C")
    slow = phrase(chord, _beat(direction="right", intensity=0.0), vibe=0.0, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    fast = phrase(chord, _beat(direction="right", intensity=1.0), vibe=0.0, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    slow_gap = slow[1].offset - slow[0].offset
    fast_gap = fast[1].offset - fast[0].offset
    assert fast_gap < slow_gap


def test_down_direction_has_bass_octaves_and_boosted_velocity() -> None:
    chord = parse_chord("C")
    left_beat = _beat(direction="left", intensity=0.5)
    down_beat = _beat(direction="down", intensity=0.5)
    left_block = phrase(chord, left_beat, vibe=0.5, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    down = phrase(chord, down_beat, vibe=0.5, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    down_notes = {e.note for e in down}
    assert midi(0, 2) in down_notes  # root octave 2
    assert midi(0, 3) in down_notes  # root octave 3
    assert min(e.velocity for e in down) > min(e.velocity for e in left_block)


def test_up_direction_runs_up_the_chord_scale() -> None:
    chord = parse_chord("C")
    events = phrase(chord, _beat(direction="up", intensity=0.5), vibe=0.2, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    notes = [e.note for e in events]
    assert notes == sorted(notes)
    assert notes[-1] - notes[0] <= 19  # ~1.5 octaves
    velocities = [e.velocity for e in events]
    assert velocities == sorted(velocities)  # rising
    scale = set(chord_scale(chord))
    for n in notes:
        assert n % 12 in scale


def test_closed_beat_is_staccato_open_is_sustain() -> None:
    chord = parse_chord("C")
    closed = phrase(chord, _beat(direction="left", closed=True), vibe=0.2, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    open_ = phrase(chord, _beat(direction="left", closed=False), vibe=0.2, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    assert all(e.duration == CFG.piano_staccato for e in closed)
    assert all(e.duration == CFG.piano_sustain for e in open_)


def test_channel_is_piano() -> None:
    chord = parse_chord("C")
    events = phrase(chord, _beat(direction="left"), vibe=0.2, tempo=None, voicer=PianoVoicer(), cfg=CFG)
    for e in events:
        assert e.channel == 1  # CH_PIANO


def test_accompaniment_bar_length() -> None:
    chord = parse_chord("C")
    tempo = 100.0
    events = accompaniment(chord, tempo, vibe=0.3, voicer=PianoVoicer(), cfg=CFG)
    bar_length = 4 * 60 / tempo
    assert events
    for e in events:
        assert e.offset < bar_length
        assert e.offset + e.duration <= bar_length + 1e-9


def test_accompaniment_bass_on_one_and_three_chord_on_two_and_four() -> None:
    chord = parse_chord("C")
    tempo = 120.0
    events = accompaniment(chord, tempo, vibe=0.3, voicer=PianoVoicer(), cfg=CFG)
    beat_dur = 60 / tempo
    offsets = sorted({round(e.offset, 6) for e in events})
    assert 0.0 in offsets
    assert round(2 * beat_dur, 6) in offsets
    assert round(beat_dur, 6) in offsets
    assert round(3 * beat_dur, 6) in offsets


def test_accompaniment_denser_when_vibe_high() -> None:
    chord = parse_chord("C")
    tempo = 120.0
    calm = accompaniment(chord, tempo, vibe=0.1, voicer=PianoVoicer(), cfg=CFG)
    hype = accompaniment(chord, tempo, vibe=0.9, voicer=PianoVoicer(), cfg=CFG)
    assert len(hype) > len(calm)
