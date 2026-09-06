from zerohero.config import CH_PIANO, MusicConfig, PianoConfig
from zerohero.music import piano
from zerohero.music.theory import parse_chord

CFG = MusicConfig()
PCFG = PianoConfig()


def pcs(notes):
    return {n % 12 for n in notes}


def test_key_at_spans_keyboard_left_low_right_high():
    assert piano.key_at(0.0, PCFG) == PCFG.key_low
    assert piano.key_at(1.0, PCFG) == PCFG.key_high
    assert piano.key_at(0.25, PCFG) < piano.key_at(0.75, PCFG)


def test_chord_tones_thicken_with_density():
    c = parse_chord("C")
    assert pcs(piano.chord_tones(c, 0.0)) == {0, 4, 7}
    assert 11 in piano.chord_tones(c, 0.5)  # maj7
    assert 2 in piano.chord_tones(c, 0.9)  # 9th
    g7 = piano.chord_tones(parse_chord("G7"), 0.5)
    assert 5 in g7 and 6 not in g7  # b7 (F) stays b7 on a dominant, no maj7


def test_voicing_at_root_on_bottom_within_octave_below_center():
    c = parse_chord("Am")
    notes = piano.voicing_at(c, 60, 0.0)
    assert notes[0] % 12 == 9
    assert 48 <= notes[0] <= 60
    assert max(notes) - min(notes) < 12
    assert pcs(notes) == {9, 0, 4}


def test_voicing_at_slash_bass_on_bottom():
    notes = piano.voicing_at(parse_chord("C/G"), 60, 0.0)
    assert notes[0] % 12 == 7


def test_grip_chord_position_intensity_and_downward_weight():
    c = parse_chord("F")
    low = piano.grip_chord(c, 0.1, 0.5, 0.2, False, CFG, PCFG)
    high = piano.grip_chord(c, 0.9, 0.5, 0.2, False, CFG, PCFG)
    assert max(e.note for e in low) < min(e.note for e in high)
    assert all(e.channel == CH_PIANO for e in low)
    soft = piano.grip_chord(c, 0.5, 0.1, 0.0, False, CFG, PCFG)
    hard = piano.grip_chord(c, 0.5, 1.0, 1.0, False, CFG, PCFG)
    assert soft[0].velocity < hard[0].velocity
    down = piano.grip_chord(c, 0.5, 0.5, 0.2, True, CFG, PCFG)
    flat = piano.grip_chord(c, 0.5, 0.5, 0.2, False, CFG, PCFG)
    assert min(e.note for e in down) == min(e.note for e in flat) - 12
    assert down[0].velocity > flat[0].velocity
    # rolled, not simultaneous, but tight
    assert 0 < flat[-1].offset < 0.06
    assert all(e.duration > 2.0 for e in flat)  # held; the mode releases early


def test_lattice_is_ascending_chord_tones_over_the_span():
    notes = piano.lattice(parse_chord("G"), 0.0, PCFG)
    assert notes == sorted(notes)
    assert pcs(notes) == {7, 11, 2}
    assert notes[0] >= PCFG.key_low and notes[-1] <= PCFG.key_high
    assert len(notes) >= 12


def test_slot_at_tracks_hand_position():
    notes = piano.lattice(parse_chord("C"), 0.0, PCFG)
    assert piano.slot_at(0.0, notes, PCFG) == 0
    assert piano.slot_at(1.0, notes, PCFG) == len(notes) - 1
    assert piano.slot_at(0.3, notes, PCFG) < piano.slot_at(0.6, notes, PCFG)


def test_sweep_note_speed_sets_velocity_and_shortens_duration():
    notes = piano.lattice(parse_chord("C"), 0.0, PCFG)
    slow = piano.sweep_note(notes, 5, 1.0, 0.2, CFG, PCFG)
    fast = piano.sweep_note(notes, 5, 9.0, 0.2, CFG, PCFG)
    assert slow.note == fast.note == notes[5]
    assert fast.velocity > slow.velocity
    assert fast.duration < slow.duration
    assert piano.sweep_note(notes, 999, 1.0, 0.0, CFG, PCFG).note == notes[-1]  # clamped


def test_passage_runs_through_scale_in_direction():
    c = parse_chord("Am")
    up = piano.passage(c, 0.5, "up", 0.5, 0.3, False, CFG, PCFG)
    down = piano.passage(c, 0.5, "down", 0.5, 0.3, False, CFG, PCFG)
    scale = set(__import__("zerohero.music.theory", fromlist=["chord_scale"]).chord_scale(c))
    assert [e.note for e in up] == sorted(e.note for e in up)
    assert [e.note for e in down] == sorted((e.note for e in down), reverse=True)
    assert all(e.note % 12 in scale for e in up + down)
    assert all(e.note > piano.key_at(0.5, PCFG) for e in up)
    offsets = [e.offset for e in up]
    assert offsets == sorted(offsets) and offsets[0] == 0.0
    # crescendo
    assert up[-1].velocity >= up[0].velocity


def test_passage_longer_and_faster_with_intensity_and_zigzag_when_erratic():
    c = parse_chord("D")
    gentle = piano.passage(c, 0.4, "up", 0.1, 0.0, False, CFG, PCFG)
    wild = piano.passage(c, 0.4, "up", 1.0, 1.0, False, CFG, PCFG)
    assert len(wild) > len(gentle)
    assert wild[1].offset < gentle[1].offset
    zig = piano.passage(c, 0.4, "up", 0.8, 0.5, True, CFG, PCFG)
    notes = [e.note for e in zig]
    assert notes != sorted(notes)  # wobbles
    assert notes[-1] > notes[0]  # but still heads up
