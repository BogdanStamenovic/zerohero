"""Timed NoteEvent player, running on its own thread. See ARCHITECTURE.md."""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from zerohero.config import CH_GUITAR, CH_PIANO
from zerohero.events import NoteEvent
from zerohero.synth.base import Synth


@dataclass(order=True)
class _Task:
    time: float
    seq: int
    channel: int = field(compare=False)
    action: Callable[[], None] = field(compare=False)
    note: int | None = field(default=None, compare=False)


class Scheduler:
    """Background thread firing note-on/note-off calls at their due times.

    A `threading.Condition` doubles as the lock guarding the heap: the thread
    waits with a timeout equal to exactly the next due event (or indefinitely
    when the heap is empty), and any call that adds or removes work notifies
    it, so there is no polling.
    """

    def __init__(self, synth: Synth) -> None:
        self._synth = synth
        self._heap: list[_Task] = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="zerohero-scheduler")
        self._thread.start()

    def play(self, events: list[NoteEvent], t0: float | None = None) -> None:
        """Schedule note-ons at t0+offset and note-offs at t0+offset+duration.
        t0 defaults to now. Events with offset 0 and no explicit t0 fire their
        note-on synchronously from the calling thread instead of the heap --
        skipping a thread wake-up shaves the only latency a scheduler adds.
        """
        now = time.monotonic()
        use_now = t0 is None
        base = t0 if t0 is not None else now
        fire_now: list[tuple[int, int, int]] = []
        with self._cond:
            for ev in events:
                on_time = base + ev.offset
                off_time = on_time + ev.duration
                if use_now and ev.offset == 0:
                    fire_now.append((ev.channel, ev.note, ev.velocity))
                else:
                    self._push(on_time, ev.channel, _note_on(self._synth, ev.channel, ev.note, ev.velocity), ev.note)
                self._push(off_time, ev.channel, _note_off(self._synth, ev.channel, ev.note), ev.note)
            self._cond.notify_all()
        for channel, note, velocity in fire_now:
            self._synth.note_on(channel, note, velocity)

    def cut(self, channel: int) -> None:
        """Cancel pending note-ons/offs on `channel` and silence it immediately."""
        with self._cond:
            self._heap = [t for t in self._heap if t.channel != channel]
            heapq.heapify(self._heap)
            self._cond.notify_all()
        self._synth.all_notes_off(channel)

    def release(self, channel: int, notes: list[int]) -> None:
        """Note-off these notes now and drop their pending note-offs; other notes keep ringing."""
        wanted = set(notes)
        with self._cond:
            self._heap = [t for t in self._heap if not (t.channel == channel and t.note in wanted)]
            heapq.heapify(self._heap)
            self._cond.notify_all()
        for n in notes:
            self._synth.note_off(channel, n)

    def panic(self) -> None:
        """Cancel everything pending and silence both channels."""
        with self._cond:
            self._heap.clear()
            self._cond.notify_all()
        self._synth.all_notes_off(CH_GUITAR)
        self._synth.all_notes_off(CH_PIANO)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join()

    def _push(self, when: float, channel: int, action: Callable[[], None], note: int | None = None) -> None:
        heapq.heappush(self._heap, _Task(when, next(self._seq), channel, action, note))

    def _run(self) -> None:
        with self._cond:
            while not self._stopped:
                now = time.monotonic()
                if not self._heap:
                    self._cond.wait()
                    continue
                due = self._heap[0].time
                if due > now:
                    self._cond.wait(timeout=due - now)
                    continue
                task = heapq.heappop(self._heap)
                # Run the action with the lock released: it calls into the synth,
                # which must never be blocked on the scheduler's own lock.
                self._cond.release()
                try:
                    task.action()
                finally:
                    self._cond.acquire()


def _note_on(synth: Synth, channel: int, note: int, velocity: int) -> Callable[[], None]:
    return lambda: synth.note_on(channel, note, velocity)


def _note_off(synth: Synth, channel: int, note: int) -> Callable[[], None]:
    return lambda: synth.note_off(channel, note)
