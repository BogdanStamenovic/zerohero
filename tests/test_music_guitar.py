import pytest

from zerohero.config import MusicConfig
from zerohero.music.guitar import STANDARD_TUNING, describe, strum, voicing
from zerohero.music.theory import parse_chord

OPEN_CHORD_SYMBOLS = [
    "C",
    "D",
    "Dm",
    "D7",
    "E",
    "Em",
    "E7",
    "F",
    "G",
    "G7",
    "A",
    "Am",
    "A7",
    "B7",
    "Bm",
    "Cmaj7",
    "Dmaj7",
    "Amaj7",
    "Em7",
    "Am7",
    "Dsus2",
    "Dsus4",
    "Asus2",
    "Asus4",
    "Esus4",
    "Cadd9",
    "Fmaj7",
]

EXPECTED_SHAPES = {
    "C": "x32010",
    "D": "xx0232",
    "Dm": "xx0231",
    "D7": "xx0212",
    "E": "022100",
    "Em": "022000",
    "E7": "020100",
    "F": "133211",
    "G": "320003",
    "G7": "320001",
    "A": "x02220",
    "Am": "x02210",
    "A7": "x02020",
    "B7": "x21202",
    "Bm": "x24432",
    "Cmaj7": "x32000",
    "Dmaj7": "xx0222",
    "Amaj7": "x02120",
    "Em7": "020000",
    "Am7": "x02010",
    "Dsus2": "xx0230",
    "Dsus4": "xx0233",
    "Asus2": "x02200",
    "Asus4": "x02230",
    "Esus4": "002200",
    "Cadd9": "x32030",
    "Fmaj7": "xx3210",
}


def test_standard_tuning() -> None:
    assert STANDARD_TUNING == (40, 45, 50, 55, 59, 64)


@pytest.mark.parametrize("symbol", OPEN_CHORD_SYMBOLS)
def test_open_shape_matches_table(symbol: str) -> None:
    chord = parse_chord(symbol)
    fret_str = describe(chord).split()[-1]
    assert fret_str == EXPECTED_SHAPES[symbol]


@pytest.mark.parametrize("symbol", OPEN_CHORD_SYMBOLS)
def test_open_shape_voicing_is_only_chord_tones(symbol: str) -> None:
    chord = parse_chord(symbol)
    notes = voicing(chord)
    pcs = set(chord.pitch_classes())
    sounding = [n for n in notes if n is not None]
    assert len(sounding) >= 3
    for n in sounding:
        assert n % 12 in pcs


@pytest.mark.parametrize("symbol", ["F#m", "Bbm7", "C#7", "C#", "Dbm"])
def test_barre_fallback_sounds_at_least_four_notes_and_only_chord_tones(symbol: str) -> None:
    chord = parse_chord(symbol)
    notes = voicing(chord)
    sounding = [n for n in notes if n is not None]
    assert len(sounding) >= 4
    pcs = set(chord.pitch_classes())
    for n in sounding:
        assert n % 12 in pcs


@pytest.mark.parametrize("symbol", ["Cdim", "Caug", "Csus2", "C6", "Cm6", "C9", "Cm9", "C5", "Adim7", "Bm7b5"])
def test_generic_voicing_is_only_chord_tones_and_at_least_three_notes(symbol: str) -> None:
    chord = parse_chord(symbol)
    notes = voicing(chord)
    sounding = [n for n in notes if n is not None]
    assert len(sounding) >= 3
    pcs = set(chord.pitch_classes())
    for n in sounding:
        assert n % 12 in pcs


def test_slash_bass_puts_bass_pitch_class_on_lowest_string() -> None:
    chord = parse_chord("C/G")
    notes = voicing(chord)
    assert notes[0] is not None
    assert notes[0] % 12 == 7  # G


def test_slash_bass_am_over_e() -> None:
    chord = parse_chord("Am/E")
    notes = voicing(chord)
    assert notes[0] is not None
    assert notes[0] % 12 == 4  # E


def test_down_strum_goes_low_to_high() -> None:
    chord = parse_chord("Am")
    events = strum(chord, "down", 0.7, MusicConfig())
    notes = [e.note for e in events]
    assert notes == sorted(notes)


def test_up_strum_goes_high_to_low() -> None:
    chord = parse_chord("Am")
    events = strum(chord, "up", 0.9, MusicConfig())
    notes = [e.note for e in events]
    assert notes == sorted(notes, reverse=True)


def test_up_strum_skips_low_strings_at_low_intensity() -> None:
    chord = parse_chord("G")  # all 6 strings sound
    down_full = strum(chord, "down", 0.9, MusicConfig())
    up_soft = strum(chord, "up", 0.1, MusicConfig())
    assert len(up_soft) < len(down_full)
    # the lowest (physically lowest-pitched) sounding string should be dropped
    lowest_note = min(e.note for e in down_full)
    assert lowest_note not in [e.note for e in up_soft]


def test_strum_velocity_in_range() -> None:
    chord = parse_chord("C")
    cfg = MusicConfig()
    for intensity in (0.0, 0.3, 0.7, 1.0):
        for direction in ("down", "up"):
            events = strum(chord, direction, intensity, cfg)
            for e in events:
                assert 1 <= e.velocity <= 127
                assert cfg.velocity_min - 1 <= e.velocity <= cfg.velocity_max + 15


def test_strum_spacing_tightens_with_intensity() -> None:
    chord = parse_chord("G")
    cfg = MusicConfig()
    slow = strum(chord, "down", 0.0, cfg)
    fast = strum(chord, "down", 1.0, cfg)
    slow_gap = slow[1].offset - slow[0].offset
    fast_gap = fast[1].offset - fast[0].offset
    assert fast_gap < slow_gap


def test_strum_duration_and_channel_from_config() -> None:
    chord = parse_chord("C")
    cfg = MusicConfig()
    events = strum(chord, "down", 0.5, cfg)
    for e in events:
        assert e.duration == cfg.guitar_sustain
        assert e.channel == 0  # CH_GUITAR


def test_describe_format() -> None:
    chord = parse_chord("Am")
    assert describe(chord) == "Am  x02210"
