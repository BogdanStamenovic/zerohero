import pytest

from zerohero.music.theory import (
    NOTE_NAMES,
    Chord,
    ChordError,
    chord_scale,
    midi,
    note_name,
    parse_chord,
    parse_progression,
)

MAJOR = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@pytest.mark.parametrize(
    "symbol,root,quality",
    [
        ("C", 0, "maj"),
        ("Cmaj", 0, "maj"),
        ("CM", 0, "maj"),
        ("Am", 9, "m"),
        ("Amin", 9, "m"),
        ("A-", 9, "m"),
        ("C7", 0, "7"),
        ("Cmaj7", 0, "maj7"),
        ("CM7", 0, "maj7"),
        ("CΔ7", 0, "maj7"),
        ("Am7", 9, "m7"),
        ("Cdim", 0, "dim"),
        ("C°", 0, "dim"),
        ("Cdim7", 0, "dim7"),
        ("Am7b5", 9, "m7b5"),
        ("Aø", 9, "m7b5"),
        ("Caug", 0, "aug"),
        ("C+", 0, "aug"),
        ("Csus2", 0, "sus2"),
        ("Csus4", 0, "sus4"),
        ("Cadd9", 0, "add9"),
        ("C6", 0, "6"),
        ("Cm6", 0, "m6"),
        ("C9", 0, "9"),
        ("Cm9", 0, "m9"),
        ("Cmaj9", 0, "maj9"),
        ("C5", 0, "5"),
    ],
)
def test_parse_every_quality(symbol: str, root: int, quality: str) -> None:
    chord = parse_chord(symbol)
    assert chord.root == root
    assert chord.quality == quality
    assert chord.bass is None


def test_sharp_and_flat_roots() -> None:
    assert parse_chord("C#").root == 1
    assert parse_chord("Db").root == 1
    assert parse_chord("F#m7").root == 6
    assert parse_chord("Bbm7").root == 10


def test_bb_vs_b_are_different_pitch_classes() -> None:
    assert parse_chord("Bb").root == 10
    assert parse_chord("B").root == 11


def test_slash_bass() -> None:
    c = parse_chord("C/G")
    assert c.root == 0
    assert c.quality == "maj"
    assert c.bass == 7
    assert c.symbol == "C/G"

    am_e = parse_chord("Am/E")
    assert am_e.root == 9
    assert am_e.bass == 4


def test_pitch_classes_include_slash_bass() -> None:
    c = parse_chord("C/G")
    assert set(c.pitch_classes()) == {0, 4, 7}  # G is already a chord tone
    d = parse_chord("D/C")  # C is not a D-major chord tone
    assert set(d.pitch_classes()) == {2, 6, 9, 0}


def test_canonical_symbol_normalises_aliases_but_keeps_root_spelling() -> None:
    assert parse_chord("Amin").symbol == "Am"
    assert parse_chord("CM7").symbol == "Cmaj7"
    assert parse_chord("Bbmin").symbol == "Bbm"
    assert str(parse_chord("A-")) == "Am"
    assert parse_chord("A-").name == "Am"


@pytest.mark.parametrize(
    "garbage,expected_token",
    [
        ("H", "H"),
        ("Xm", "X"),
        ("Cxyz", "xyz"),
        ("C#bogus", "bogus"),
    ],
)
def test_garbage_raises_chorderror_naming_token(garbage: str, expected_token: str) -> None:
    with pytest.raises(ChordError, match=expected_token):
        parse_chord(garbage)


def test_bad_slash_bass_names_the_token() -> None:
    with pytest.raises(ChordError, match="Z"):
        parse_chord("C/Z")


def test_empty_chord_raises() -> None:
    with pytest.raises(ChordError):
        parse_chord("")
    with pytest.raises(ChordError):
        parse_chord("   ")


def test_progression_splits_on_whitespace_commas_pipes() -> None:
    chords = parse_progression("Am G | C F")
    assert [c.symbol for c in chords] == ["Am", "G", "C", "F"]

    chords2 = parse_progression("Am,G,C,F")
    assert [c.symbol for c in chords2] == ["Am", "G", "C", "F"]


def test_progression_dash_as_separator_vs_minor_suffix() -> None:
    # "Am-G" is two chords; "A-" is A minor (dash directly on the root).
    assert [c.symbol for c in parse_progression("Am-G")] == ["Am", "G"]
    assert [c.symbol for c in parse_progression("A- G")] == ["Am", "G"]
    assert [c.symbol for c in parse_progression("A-")] == ["Am"]


def test_progression_empty_raises() -> None:
    with pytest.raises(ChordError):
        parse_progression("")
    with pytest.raises(ChordError):
        parse_progression("   ")


def test_midi_c4_is_60() -> None:
    assert midi(0, 4) == 60


def test_note_name_roundtrip() -> None:
    assert note_name(60) == "C4"
    assert note_name(61) == "C#4"
    assert note_name(69) == "A4"
    for pc, name in enumerate(NOTE_NAMES):
        assert note_name(midi(pc, 4)).startswith(name)


def test_chord_scale_is_pitch_classes_rooted_at_chord() -> None:
    c_major_scale = chord_scale(parse_chord("C"))
    assert set(c_major_scale) == {0, 2, 4, 5, 7, 9, 11}  # C ionian

    a_minor_scale = chord_scale(parse_chord("Am"))
    assert set(a_minor_scale) == {9, 11, 0, 2, 4, 5, 7}  # A aeolian

    g7_scale = chord_scale(parse_chord("G7"))
    assert set(g7_scale) == {(7 + iv) % 12 for iv in (0, 2, 4, 5, 7, 9, 10)}  # G mixolydian


def test_chord_is_frozen_and_hashable() -> None:
    c = parse_chord("Am7")
    assert isinstance(c, Chord)
    with pytest.raises(AttributeError):  # frozen dataclass -> FrozenInstanceError
        c.root = 0  # type: ignore[misc]
    hash(c)  # must not raise
