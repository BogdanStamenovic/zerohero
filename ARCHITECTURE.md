# zerohero architecture

Air guitar and air piano driven by a webcam. You type a chord progression,
the camera watches your hands, and gestures become notes.

This file is the contract between modules. Anyone touching a module reads the
section for it and the sections for its neighbours. Keep it current.

## Pipeline

```
camera ──► HandTracker ──► GesturePipeline ──► Mode (guitar | piano) ──► Scheduler ──► Synth ──► audio
 (thread)   (MediaPipe)     (state machines)    (music theory)           (thread)   (fluidsynth
                                   │                 ▲                                 or numpy)
                                   ▼                 │
                                Overlay          Link (lead/follow over TCP or RFCOMM)
```

One main loop, two helper threads:

- **Capture thread** (`vision/camera.py`) reads frames continuously and keeps
  only the latest one. Frames are dropped, never queued, so latency does not
  build up when inference is slower than the camera.
- **Scheduler thread** (`synth/scheduler.py`) fires timed `NoteEvent`s
  (strum spread, arpeggios, note-offs) so the main loop never sleeps.

Everything else runs on the main loop, per frame: detect hands, update gesture
state machines, hand events to the active mode, draw the overlay.

### Latency budget (measured on the dev laptop, i7-1265U, no GPU)

| stage | ms |
|---|---|
| camera read (640x480 MJPG) | 12 |
| hand landmarker (lite, CPU) | 40 |
| gesture + music + scheduling | < 1 |
| fluidsynth to PulseAudio | ~10 |

Roughly 60-80 ms from gesture onset to sound. Bluetooth audio adds 100-200 ms
on top; use wired output when playing.

## Coordinate conventions

- The camera image is **mirrored** (`cv2.flip(frame, 1)`) before tracking so
  the window looks like a mirror. `Hand.side` is assigned by **position**,
  not by MediaPipe's handedness label: that label is the chirality of the 2D
  projection and flips when the user shows the back of the hand, which a
  strumming or conducting hand does constantly. Two hands: smaller x is
  "left". One hand: keeps the side it had last frame (nearest previous
  palm), else splits at the centre. `--no-mirror` only turns the flip off.
- Hand shape comes from the gesture recognizer's label (`Hand.gesture`:
  Closed_Fist, Open_Palm, ...), not from landmark geometry, which cannot
  tell a flat hand pointing at the camera from a fist. `closed_score` is
  1 for a confident fist, 0 for any other confident label, 0.5 when unsure
  so detector hysteresis holds.
- Landmark `x`, `y` are normalised to `0..1`, `y` grows downward.
  After mirroring, `+x` is the user's right.
  "Hit toward the left" means palm velocity `vx < 0`.
- Velocities are in **hand widths per second**, where hand width is the
  distance wrist (0) to middle-finger MCP (9). This makes thresholds
  independent of distance to the camera.
- Time `t` is `time.monotonic()` seconds everywhere. Replay files store it too.

## Modules

```
src/zerohero/
  cli.py            argparse entry point (subcommands below)
  config.py         Config dataclass, all tunables, TOML override
  events.py         shared dataclasses: Landmark, Hand, Frame, gesture events, NoteEvent
  paths.py          data dir, asset locations
  assets.py         download/verify hand model + soundfont
  music/theory.py   chord symbol parser, pitch classes
  music/guitar.py   guitar voicings, strum -> NoteEvents
  music/piano.py    piano voicings with voice leading, Beat -> phrase NoteEvents
  synth/base.py     Synth protocol
  synth/fluid.py    fluidsynth backend
  synth/basic.py    numpy + sounddevice fallback (Karplus-Strong, additive piano)
  synth/scheduler.py timed NoteEvent player
  vision/camera.py  capture thread
  vision/hands.py   MediaPipe gesture recognizer wrapper -> Frame (landmarks + fist/palm label)
  vision/replay.py  JSONL record + replay of Frames (no camera needed)
  vision/source.py  FrameSource protocol, CameraSource, ReplaySource
  gestures/features.py  per-hand features from landmarks
  gestures/track.py     per-side history, smoothed velocity
  gestures/strum.py     StrumDetector
  gestures/fist.py      FistDetector
  gestures/conduct.py   BeatDetector (conducting hits)
  gestures/vibe.py      VibeMeter (energy 0..1)
  gestures/pipeline.py  GesturePipeline: Frame -> list[GestureEvent]
  gestures/pianist.py   per-hand piano detectors: grip hit, sweep, passage
  engine/session.py     progression state
  engine/guitar_mode.py
  engine/piano_mode.py
  engine/app.py         wires everything, main loop
  link/protocol.py      message types, JSON-lines codec
  link/transport.py     stream socket wrapper, TCP + RFCOMM listen/connect
  link/discovery.py     UDP broadcast beacon + discover
  link/lead.py          Lead: accept followers, broadcast events
  link/follow.py        Follower: connect, clock offset, tempo estimate
  ui/overlay.py         OpenCV window: landmarks, chord strip, meters; key handling
  ui/terminal.py        the same view drawn with braille dots in the terminal
  ui/keys.py            cross-platform non-blocking key reader (termios / msvcrt)
  ui/view.py            View protocol and open_view(): window | terminal | none
```

### events.py (shared, read this first)

See the file. Summary:

- `Hand(side, score, landmarks[21], world[21] | None)`
- `Frame(t, width, height, hands, image | None)`
- Gesture events, all frozen dataclasses with `t` and `hand`:
  - `Strum(direction: "down"|"up", intensity: 0..1)` from the pick hand
  - `FistClose`, `FistOpen` edge events
  - `Beat(direction, intensity, closed)` a generic conducting hit (unused by
    the current piano; kept for experiments)
  - pianist events `GripHit`, `GripRelease`, `SweepStep`, `Passage` (piano)
- `NoteEvent(offset, note, velocity, duration, channel)`; `offset` is seconds
  relative to when the phrase is scheduled.

MIDI channels: `CH_GUITAR = 0`, `CH_PIANO = 1` (in `config.py`).

### music

`parse_chord("F#m7")` -> `Chord(root=6, quality="m7", bass=None, symbol=...)`.
Accepted: major (`C`), minor (`Am`, `Amin`, `A-`), `7`, `maj7`, `m7`, `dim`,
`dim7`, `m7b5`, `aug`, `sus2`, `sus4`, `add9`, `6`, `m6`, `9`, `5`, slash
bass (`C/G`), flats and sharps. Unknown symbol raises `ChordError` with the
offending token. `parse_progression("Am G | C F")` splits on whitespace,
commas, pipes and dashes.

`guitar.voicing(chord)` returns six entries low E to high e, MIDI note or
`None` for a muted string, from a shape table of open chords with a movable
barre fallback (E-shape and A-shape). `guitar.strum(chord, direction,
intensity)` returns `NoteEvent`s: down strums go low to high, up strums high
to low, string spacing scales inversely with intensity (fast strum = tight).
Velocity is `40 + 87 * intensity`. Previous chord notes are cut when a new
strum starts (the mode handles that via `Scheduler.cut(channel)`).

### piano

The piano is accompaniment to a guitar session: the chord comes from the
linked guitar and gestures never change it. The hand is the pianist.
`gestures/pianist.py` has one `Pianist` per hand fed every frame from
`PianoMode.observe()` with the hand's track and its slot on the lattice:

| hand | event | plays |
|---|---|---|
| gripped (fingers curled, `closed_score` hysteresis 0.55/0.35) and a sharp onset | `GripHit(x, intensity, downward)` | `piano.grip_chord`: closed voicing with the root just below the key at `x`, held until `GripRelease` (`Scheduler.release`) |
| open, moving sideways above `sweep_speed_on`, crossing lattice slots | `SweepStep(slot, direction, speed)` | `piano.sweep_note`: the chord tone at that slot; speed sets velocity and shortens the note |
| open, sharp vertical onset, or erratic (>= 2 horizontal reversals in 0.5 s) | `Passage(x, direction, intensity, erratic)` | `piano.passage`: a run through `chord_scale` from the key at `x`, zigzag when erratic |

"Where on the keyboard" is the hand's x mapped onto MIDI `key_low..key_high`
(C2..C6), same map for both hands. The lattice is the chord tones tiled
across that span; `vibe` thickens it with 7th and 9th in three buckets so it
does not flicker. Accents come from the predicted peak speed of the hit,
dynamics from `vibe`. Tunables live in `PianoConfig`.

### synth

`Synth` protocol: `note_on(ch, note, vel)`, `note_off(ch, note)`,
`program(ch, program)`, `all_notes_off(ch)`, `close()`.
`open_synth(config)` tries fluidsynth with the configured soundfont, then falls
back to `basic.NumpySynth` with a warning. Programs: guitar 25 (steel), piano 0.

`Scheduler(synth)`: `play(events, t0=None)` schedules note-ons at
`t0 + offset` and note-offs at `t0 + offset + duration`; `cut(channel)` cancels
pending events on a channel and sends all-notes-off; `release(channel, notes)`
note-offs just those notes and drops their pending offs (held chords);
`stop()`.

### vision

`Camera(index, width, height, fps)`: `.start()`, `.latest() -> (t, bgr) | None`,
`.stop()`. `HandTracker(model_path, max_hands=2, mirror=True)`:
`.process(bgr, t) -> Frame`. `ReplaySource(path, realtime=True)` yields the
same `Frame`s a camera would, minus `image`, so the whole pipeline runs
without a camera in tests and for tuning. `--record path` writes that file.

### gestures

`HandFeatures.from_hand(hand)`: palm centre (mean of wrist and the four
finger MCPs), hand width, per-finger extended flags (tip farther from wrist
than PIP is), `closed_score` in `0..1` (fraction of the four fingers curled).

`HandTrack` per side keeps a short history and an exponentially smoothed
velocity in hand-widths/s. Tracks time out after `lost_after` seconds without
a detection so a hand leaving the frame does not produce a phantom strum when
it returns.

Detectors are onset based: fire when speed crosses the high threshold, then
lock until it drops under the low threshold (hysteresis) and a refractory
period has passed. Onset firing gives the lowest latency. Intensity is the
onset speed mapped into `0..1` between the threshold and `speed_full`.

- `StrumDetector(side)`: vertical component only; direction from sign of `vy`.
- `FistDetector(side)`: `closed_score` with hysteresis (close above 0.7, open
  below 0.4), edge events, minimum hold to reject flicker.
- `BeatDetector(sides)`: 2D; direction is the dominant axis of the onset
  velocity; `closed` from the fist state at that instant.
- `VibeMeter`: rolling RMS of both hands' speed over ~1.5 s, mapped to `0..1`.

`GesturePipeline(config).update(frame) -> list[GestureEvent]` runs all of them
and exposes `.vibe` and `.tracks` for the overlay.

### engine

`Session(progression)`: `current`, `index`, `next()`, `prev()`, `goto(i)`,
wraps around. `GuitarMode`: `Strum` -> cut guitar channel, play strum of
`session.current`; `FistClose` on the fret hand -> `session.next()`. Both are
forwarded to the `Lead` link if attached. `PianoMode`: chord from the link
(`--follow`), never from gestures; `observe(pipeline, frame)` runs the two
`Pianist` detectors each frame and plays what they emit; `n`/`p` step chords
standalone for testing. `app.run(config)` builds the
graph and runs the loop; `q` quits, `n`/`p` step chords, space is a manual
strum, `r` resets.

### link

Wire format is one JSON object per line, UTF-8, over any stream socket, so TCP
and Bluetooth RFCOMM share all code above the socket. Messages:

```
{"type":"hello","role":"lead"|"follow","name":str,"version":1}
{"type":"chord","index":int,"symbol":str,"t":float}      lead time
{"type":"strum","direction":str,"intensity":float,"t":float}
{"type":"ping","t":float} / {"type":"pong","t":float,"rt":float}
{"type":"bye"}
```

Clock: follower sends 5 pings on connect and periodically; offset is the
median of `(pong.rt - (t_send + t_recv)/2)`. Lead timestamps are converted to
follower monotonic time with that offset. Tempo: follower keeps the last 8
strum onsets, tempo = 60 / median inter-onset interval, dropped after 3 s
without strums.

Discovery: lead broadcasts `{"zerohero":1,"port":P,"name":...}` on UDP
`255.255.255.255:47474` once a second; `--follow auto` listens for up to 5 s.
Direct `--follow host[:port]` skips discovery and works over Tailscale.
Bluetooth: `--lead --transport bt` listens on RFCOMM channel 3;
`--follow bt:AA:BB:CC:DD:EE:FF` connects. Needs the devices paired first. If
the Python build has `AF_BLUETOOTH` the socket module is used directly;
otherwise (uv's python-build-standalone, which install.py prefers, lacks it)
`link/bt_raw.py` makes the socket through libc with ctypes and a hand-packed
`sockaddr_rc`, doing bind/listen/accept/connect itself because CPython refuses
address operations on a family it was not built with. No SDP record is
registered, so the channel number is fixed. Listen and connect-failure are
verified on one machine; two-device operation is untested; see README.

### ui

Two interchangeable views behind one protocol (`start`, `draw(frame, state)`,
`poll_key`, `close`). `WindowView` is an OpenCV window with the camera image
and the overlay. `TerminalView` draws the hand skeletons with braille dots
(2x4 dots per cell) plus the chord strip and meters, using ANSI escapes and
the alternate screen; on Windows it enables virtual-terminal processing
through `SetConsoleMode`. Key input is `termios` on POSIX and `msvcrt` on
Windows. `--ui auto` picks the window when a display exists (always on
Windows/macOS, `DISPLAY`/`WAYLAND_DISPLAY` on Linux), else the terminal when
stdout is a tty, else nothing.

## Setup and packaging

- Python **3.12** pinned. MediaPipe 1.0.1 on Python 3.14 gets SIGKILLed on
  landmarker creation on the dev laptop; 0.10.x on 3.12 works. `install.py`
  uses `uv` to provision 3.12 when present, else expects `python3.12`.
- Assets live in `$XDG_DATA_HOME/zerohero` (default `~/.local/share/zerohero`):
  `gesture_recognizer.task` (8.4 MB, Google) and `GeneralUser-GS.sf2` (32 MB,
  GitHub mrbumpy409). `zerohero setup` downloads them; `install.py` calls it.
- System dependency: `libfluidsynth` (Arch `fluidsynth`, Debian
  `libfluidsynth3`, brew `fluid-synth`). Without it the numpy synth is used.
- `ownbox.yaml` follows the cvoice manifest style.
