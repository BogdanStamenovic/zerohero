# zerohero

Air guitar and air piano from a webcam. You type a chord progression, the
camera watches your hands, and gestures become notes. Two devices can link so
the piano follows the guitar.

No instrument, no MIDI controller, no GPU. A laptop webcam and a speaker.

## What it does

**Guitar.** Give it chords in order. Strum with your right hand and it plays
the current chord; a down-stroke strums low to high, an up-stroke high to low,
and the speed of your hand sets how hard. Make a fist with your left hand to
move to the next chord.

**Piano.** The piano goes with the guitar: it plays whatever chord the
guitar is on and never changes it. Your hands are the pianist. Where your
hand is across the frame is where you are on the keyboard, left is low,
right is high. What the hand does decides what plays:

| the hand | what plays |
|---|---|
| hit with a grip, fingers curled like holding a chord | a block chord at that spot, held until you open the hand |
| open hand sweeping left or right | an arpeggio: the chord's notes are laid across the keyboard and each one plays as your hand passes it, so your speed is the tempo and a fast sweep is a glissando |
| open hand hitting up or down, or wiggling about | a run through the chord's scale in that direction, longer and faster the harder you hit |

How hard you hit sets the accent. How much you are moving overall sets the
dynamics and how thick the chords get (7ths, 9ths). Both hands work
independently, so a left-hand chord can hold under a right-hand sweep.

**Link.** One device runs the guitar as *lead*, the other runs the piano as
*follower*. The follower's chord changes when the guitar's does. Works over
any network that can route TCP (LAN, Tailscale) and, on Linux, over
Bluetooth RFCOMM. Standalone `zerohero piano "Am"` holds a chord for
practising, with `n`/`p` to step.

## Install

With [ownbox](https://github.com/BogdanStamenovic/ownbox):

```
ownbox install BogdanStamenovic/zerohero
```

By hand:

```
git clone https://github.com/BogdanStamenovic/zerohero
cd zerohero
python3 install.py
.venv/bin/zerohero doctor
```

`install.py` creates a Python 3.12 virtualenv (through `uv` if present, which
downloads 3.12 on its own), installs the package, and downloads two assets into
`~/.local/share/zerohero`: MediaPipe's hand gesture recognizer model (8 MB,
landmarks plus fist/palm labels) and the GeneralUser GS soundfont (32 MB).

System dependency: `libfluidsynth` (Arch `fluidsynth`, Debian
`libfluidsynth3`, macOS `brew install fluid-synth`). Without it a built-in
numpy synth is used, which works but sounds like a built-in numpy synth.

## Use

```
zerohero guitar "Am G C F"
zerohero piano  "Am G C F"
zerohero chords "Am G C F"        # prints voicings and plays them, no camera
zerohero calibrate                # live readout of what the tracker sees
```

Chord symbols: `C`, `Am`, `F#m7`, `Cmaj7`, `G7`, `Dsus4`, `Bdim`, `Eaug`,
`Cadd9`, `A5`, `C/G` and so on. Separate with spaces, commas or pipes.

The hand-tracking view is an OpenCV window by default. Over SSH, in a bare
console, or with `--ui terminal`, the same view is drawn in the terminal with
braille dots: hand skeletons, velocity arrows, the chord strip and the vibe
meter. It works in Windows Terminal and any Linux terminal with a Unicode
font; `--ui none` turns the view off entirely.

Keys in the window or terminal: `q` quit, `n`/`p` next and previous chord, space is a
manual strum or hit, `r` back to the first chord.

Sit about an arm's length from the camera with both hands in frame. The
window is a mirror: your right hand is on the right. If it is not, your camera
already mirrors; pass `--no-mirror`.

### Linking two devices

On the guitar machine:

```
zerohero guitar "Am G C F" --lead
```

On the piano machine, same network:

```
zerohero piano "Am G C F" --follow auto        # finds the lead by UDP broadcast
zerohero piano "Am G C F" --follow 100.101.1.2 # direct, e.g. a Tailscale IP
```

Bluetooth RFCOMM (Linux, devices paired first, channel 3):

```
zerohero guitar "Am G C F" --lead --transport bt
zerohero piano  "Am G C F" --follow bt:AA:BB:CC:DD:EE:FF
```

The follower keeps playing on its own if the link drops and reconnects when
the lead is back.

### Options worth knowing

- `--left-handed` swaps the strum and fist hands.
- `--record file.jsonl` saves the hand landmarks; `--replay file.jsonl` runs
  the whole pipeline from that file without a camera. Useful for tuning.
- `--synth basic` forces the numpy synth, `--synth none` mutes.
- `--ui auto|window|terminal|none` chooses the view; `--no-window` is
  shorthand for the terminal view. While the terminal view is up, log lines
  go to `~/.local/state/zerohero/zerohero.log` instead of the screen.
- Thresholds live in `~/.config/zerohero/config.toml`; every field of
  `Config` in `src/zerohero/config.py` can be set there.

## Limitations, honestly

- Latency from gesture to sound is 60-80 ms on a 2022 laptop CPU. Playable,
  not tight. Bluetooth headphones add 100-200 ms more; use wired audio.
- The tracker runs at 15-20 fps on the dev laptop's CPU. Very fast strums
  (more than about 5 per second) start to merge.
- Which hand is which is decided by position, because MediaPipe's own
  left/right label flips when it sees the back of your hand. Sit centred:
  the hand further left in the mirror is your left. Crossing your hands
  confuses it.
- The Bluetooth transport has not been tested between two real devices yet.
  On the dev laptop it binds and listens on RFCOMM channel 3 and a connect
  to a bogus address fails cleanly, and that is as far as one machine gets.
  TCP has been tested end to end on loopback. The Python that `install.py`
  provisions through uv is built without Bluetooth sockets, so RFCOMM goes
  through libc via ctypes; `zerohero doctor` tells you which path you have.
- The piano has no rhythm of its own. It plays when your hand says so.
- No individual string plucking, no bends, no palm mutes. Strums only.

## What does not exist yet

- A single-device mode with both instruments.
- Other instruments (the synth is General MIDI, so this is mostly a UI
  problem).
- macOS and Windows have not been run. The code has Windows paths for the
  camera backend, key input and the terminal escapes, and the installer is
  written for all three, but nobody has pressed the button yet.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). Short version: a capture thread keeps
the newest camera frame, the main loop runs MediaPipe hand tracking and a set
of small gesture state machines, the active mode turns gestures into
`NoteEvent`s, a scheduler thread feeds them to fluidsynth. The link is
JSON lines over a stream socket, so TCP and RFCOMM share everything above the
socket.

## License

MIT.
