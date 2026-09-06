from zerohero.config import CH_PIANO, MusicConfig, PianoConfig
from zerohero.music import piano, theory
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


def test_chord_hit_position_intensity_and_weight():
    c = parse_chord("F")
    low = piano.chord_hit(c, 0.1, 0.5, 0.2, CFG, PCFG)
    high = piano.chord_hit(c, 0.9, 0.5, 0.2, CFG, PCFG)
    assert max(e.note for e in low) < min(e.note for e in high)
    assert all(e.channel == CH_PIANO for e in low)
    soft = piano.chord_hit(c, 0.5, 0.1, 0.0, CFG, PCFG)
    hard = piano.chord_hit(c, 0.5, 1.0, 1.0, CFG, PCFG)
    assert soft[0].velocity < hard[0].velocity
    assert min(e.note for e in hard) == min(e.note for e in soft) - 12  # bass weight when leaning in
    assert 0 < soft[-1].offset < 0.06  # rolled, not simultaneous
    assert all(e.duration == CFG.piano_sustain for e in soft)


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


def test_zigzag_stays_in_scale_near_the_hand_and_drifts():
    c = parse_chord("Am")
    scale = set(theory.chord_scale(c))
    z = piano.Zigzag(seed=3)
    notes = [z.next(c, 0.5, "none", 0.5, 0.3, CFG, PCFG) for _ in range(30)]
    centre = piano.key_at(0.5, PCFG)
    assert all(e.note % 12 in scale for e in notes)
    assert all(abs(e.note - centre) <= 16 for e in notes)
    pitches = [e.note for e in notes]
    assert pitches != sorted(pitches) and pitches != sorted(pitches, reverse=True)  # zigzags
    assert all(e.channel == CH_PIANO and 0 < e.velocity <= 127 for e in notes)
    # drifting right over many notes ends higher than drifting left
    zr, zl = piano.Zigzag(seed=5), piano.Zigzag(seed=5)
    right = [zr.next(c, 0.5, "right", 0.5, 0.3, CFG, PCFG).note for _ in range(40)]
    left = [zl.next(c, 0.5, "left", 0.5, 0.3, CFG, PCFG).note for _ in range(40)]
    assert sum(right[-10:]) > sum(left[-10:])
    # livelier fingers play harder
    quiet = piano.Zigzag(seed=1).next(c, 0.5, "none", 0.0, 0.0, CFG, PCFG).velocity
    lively = piano.Zigzag(seed=1).next(c, 0.5, "none", 1.0, 0.0, CFG, PCFG).velocity
    assert lively > quiet
