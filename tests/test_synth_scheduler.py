from __future__ import annotations

import time

from zerohero.config import CH_GUITAR, CH_PIANO
from zerohero.events import NoteEvent
from zerohero.synth.base import NullSynth
from zerohero.synth.scheduler import Scheduler


def test_immediate_zero_offset_note_on_fires_synchronously() -> None:
    synth = NullSynth()
    sched = Scheduler(synth)
    try:
        sched.play([NoteEvent(offset=0.0, note=60, velocity=100, duration=10.0, channel=CH_GUITAR)])
        # No heap wake-up needed: the note-on must already be recorded.
        assert synth.calls[0] == ("note_on", CH_GUITAR, 60, 100)
    finally:
        sched.stop()


def test_ordering_of_scheduled_events() -> None:
    synth = NullSynth()
    sched = Scheduler(synth)
    try:
        t0 = time.monotonic()
        events = [
            NoteEvent(offset=0.05, note=64, velocity=90, duration=0.02, channel=CH_GUITAR),
            NoteEvent(offset=0.01, note=60, velocity=90, duration=0.02, channel=CH_GUITAR),
        ]
        sched.play(events, t0=t0)
        time.sleep(0.15)
        note_on_calls = [c for c in synth.calls if c[0] == "note_on"]
        assert [c[2] for c in note_on_calls] == [60, 64]
    finally:
        sched.stop()


def test_note_off_after_duration() -> None:
    synth = NullSynth()
    sched = Scheduler(synth)
    try:
        sched.play([NoteEvent(offset=0.0, note=60, velocity=100, duration=0.05, channel=CH_GUITAR)])
        assert ("note_off", CH_GUITAR, 60) not in synth.calls
        time.sleep(0.15)
        assert ("note_off", CH_GUITAR, 60) in synth.calls
    finally:
        sched.stop()


def test_cut_cancels_pending_events_on_one_channel_only() -> None:
    synth = NullSynth()
    sched = Scheduler(synth)
    try:
        t0 = time.monotonic() + 0.2
        sched.play([NoteEvent(offset=0.0, note=60, velocity=100, duration=1.0, channel=CH_GUITAR)], t0=t0)
        sched.play([NoteEvent(offset=0.0, note=72, velocity=100, duration=1.0, channel=CH_PIANO)], t0=t0)
        sched.cut(CH_GUITAR)
        # all_notes_off fires immediately on cut.
        assert ("all_notes_off", CH_GUITAR) in synth.calls
        time.sleep(0.4)
        note_on_notes = {c[2] for c in synth.calls if c[0] == "note_on"}
        assert 60 not in note_on_notes  # cancelled before it was due
        assert 72 in note_on_notes  # other channel unaffected
    finally:
        sched.stop()


def test_panic_cuts_both_channels() -> None:
    synth = NullSynth()
    sched = Scheduler(synth)
    try:
        t0 = time.monotonic() + 0.2
        sched.play([NoteEvent(offset=0.0, note=60, velocity=100, duration=1.0, channel=CH_GUITAR)], t0=t0)
        sched.play([NoteEvent(offset=0.0, note=72, velocity=100, duration=1.0, channel=CH_PIANO)], t0=t0)
        sched.panic()
        time.sleep(0.4)
        note_on_notes = {c[2] for c in synth.calls if c[0] == "note_on"}
        assert note_on_notes == set()
    finally:
        sched.stop()


def test_stop_joins_thread() -> None:
    synth = NullSynth()
    sched = Scheduler(synth)
    sched.stop()
    assert not sched._thread.is_alive()
