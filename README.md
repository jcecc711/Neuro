# NOVA: real-time EEG neurofeedback

NOVA reads EEG from your headset, measures your brain rhythms in real time, and guides you
through training sessions with ethereal soundscapes that respond as your brain moves toward
its target.

> **Not a medical device.** This is for research, education and personal experimentation.
> It does not diagnose or treat any condition.

## Start here: the NOVA app

```bash
pip install -e ".[brainflow]"
nova app
```

`nova app` opens NOVA in your browser at http://127.0.0.1:8765. Choose a program, pick
your headset, press **Begin session**, and put on headphones. The app does the rest:

1. **Arrive** (about 2 min). The soundscape fades in while NOVA measures your resting levels.
2. **Warm-up.** An easy target, so the sound rewards you often while you find it.
3. **Training blocks**, separated by one-minute **Breathe** interludes. During each
   interlude the sound swells at six breaths a minute for you to follow.
4. **Integrate** (2-3 min). Rewards stop and the sound thins and fades out.

While you train, the soundscape is the feedback. When your brain meets the target, a
shimmer layer blooms, the pads open up and bells chime more often. When it drifts, the
sound stays soft and muted, and it never cuts out. The screen shows the current phase,
spoken-style guidance, your progress through the session, a live cortical map, the EEG,
spectrum, hemispheric balance and the training trace. At the end you get a summary per
phase, and a CSV log is saved in `sessions/`.

Without a headset, choose **Demo signal** to try any program with generated EEG.

### What's on screen during a session

- **3D brain** on a holographic platform, coloured by live activity at each sensor. Drag
  to turn it; zoom with the buttons or the scroll wheel.
- **On target card:** the share of training time you've been rewarded. **Set new target**
  changes difficulty mid-session: Gentle (70%), Balanced (60%) or Challenging (50%).
- **Band cards:** concentration (γ), active mind (β), calm focus (SMR), reflective (α),
  drowsy (θ) and deep rest (δ). Each shows its live peak frequency and share of total
  power at the training sites. The rhythm being trained is outlined.
- **Brain activity:** every band's live trace with its peak frequency.
- **Time in each state:** how long your brain has spent dominated by each rhythm.
- **Timeline** of the whole session, and tabs for phases, training trace, spectrum, raw
  channels, state indices and left/right balance.

### History and reports

Every finished session is saved (`sessions/*.json` next to the CSV log). **My history**
lists them with time on target per session against the target line, change from baseline
and top brain state. **Reports** shows minutes trained per day, total time in each state,
and per-program progress.

## Programs

| Program | For | Trains | Sensors | Research basis |
|---|---|---|---|---|
| Focus Builder | ADHD and attention | theta/beta down, SMR up | Cz, C4 | Most studied |
| Mood Lift | Low mood and depression | frontal alpha asymmetry | F3/F4 (or AF7/AF8) | Emerging |
| Calm Waters | Anxiety | posterior alpha up, then theta/alpha | Pz, O1, O2 | Limited |
| Unwind | Stress | SMR up, then alpha up | C4, Pz | Limited |
| Recharge | Fatigue and low energy | beta up, theta/beta down | Cz, Fz | Limited |
| Deep Rest | Deep relaxation | alpha, then alpha-theta (eyes closed) | Pz, O1, O2 | Limited |
| Peak Focus | Calm, sharp concentration | SMR, then beta | C4, Cz | Limited |

Each program shows its research basis in the app. **These are training programs, not
treatments.** They follow protocols from published neurofeedback research, and the
evidence ranges from moderate (ADHD) to preliminary. For ADHD, depression or anxiety,
work with a qualified clinician. If you are in crisis, contact local emergency services
or a crisis line (call or text 988 in the US).

Every program comes in Short, Standard and Extended lengths; this scales the training
blocks while arrival and integration keep their length.

**Custom programs.** In the app, open **Build a custom program**. Choose:
- the target rhythm, whether it should go up or down, and optionally a rhythm to measure
  it against (for example theta relative to beta),
- the sensor sites, length, number of blocks and how often to reward,
- the soundscape, eyes open or closed, and an optional binaural beat.

Programs are saved to `~/.nova/programs/` as JSON, which you can edit or share.

## Soundscapes

All sound is synthesized live in the browser, so every session sounds slightly different:
detuned pads in a long reverb, a low drone, a texture layer, and bell chimes, moving through
a slow chord progression. Each phase has its own voicing; each new phase opens on a new chord.

| Soundscape | Character |
|---|---|
| Aurora | airy Lydian pads and glass shimmer |
| Deep Ocean | low swells, slow waves, distant bells |
| Celestial | bright bells over a floating major drone |
| Forest Rain | soft rain, warm pads, wooden chimes |
| Nebula | dark, slow drones with a glowing upper layer |
| Dawn | warm, rising major chords |

You can preview any soundscape before a session, and switch soundscape or volume during one.
Programs with a binaural beat play it quietly under the soundscape (headphones required).

## Supported hardware

| Source | Choose in the app / `--device` | Examples |
|---|---|---|
| BrainFlow | the headset's name, e.g. `muse_2`, `cyton` | OpenBCI Cyton / Ganglion / Daisy, Muse 2 / S, Neurosity Crown, BrainBit, g.tec Unicorn, and others |
| Lab Streaming Layer | `lsl` | Anything that publishes an LSL EEG stream (OpenBCI GUI, Muse via muselsl, BrainVision, ...) |
| Demo signal | `demo` | No hardware. Generated EEG on a standard 10-20 montage |

Run `nova devices` to list every BrainFlow board name. Electrode positions use standard
10-20/10-10 names (`Fp1`, `C3`, `O2`, `AF7`, `TP9`, ...).

## Install

```bash
pip install -e ".[all]"        # BrainFlow, LSL, terminal audio and pygame visual
pip install -e ".[brainflow]"  # enough for the app with most headsets
```

## Command line

```bash
nova programs                                    # list programs and their phases
nova run --program calm-anxiety --device muse_2  # run a program in the terminal
nova run --program focus-adhd --length 0.6 --device cyton --serial-port /dev/ttyUSB0 \
         --feedback console,dashboard            # shorter, and watch it in the browser
nova run --program ~/my-program.json             # a program file

nova check --device cyton --serial-port /dev/ttyUSB0   # check electrode contact first
nova run --device muse_2 --protocol alpha              # a single protocol, no program
```

Use `--line-freq 50` (or "Mains power: 50 Hz" in the app) in Europe, Asia and other regions
with 50 Hz mains power.

## How rewards work

Every 250 ms the last 2 s of EEG is band-pass filtered (1-40 Hz) with a mains notch, and band
powers are computed at the program's sensor sites. You are rewarded when:
- the metric is past an **adaptive threshold**, set so you hit it a target share of the time
  (75% in warm-up, about 60% in training blocks, a little less in the final block),
- no **inhibit band** (for example, theta from eye movement or high beta from muscle
  tension) is spiking,
- the signal has no **artifact** (peak-to-peak above `--artifact-uv`),
- and all of that has held for half a second.

Thresholds are calibrated during **Arrive** and keep adapting as you train. Use
`--freeze-thresholds` to fix them when training starts.

## Protocols

| Name | Trains | Typical sites |
|---|---|---|
| `alpha` | relative alpha (8-12 Hz) up | Pz, O1, O2 |
| `smr` | relative SMR (12-15 Hz) up | C4, Cz, C3 |
| `theta-beta` | theta/beta ratio down | Cz, Fz |
| `alpha-theta` | theta/alpha ratio up | Pz, O1, O2 |
| `beta` | relative beta (15-22 Hz) up | Cz, Fz |
| `alpha-asymmetry` | frontal alpha, right minus left (log) up | F3/F4, AF7/AF8 |

To add a protocol, add a `Protocol(...)` entry in `nova/protocols.py`.

## Terminal feedback outputs

For `nova run`; the app uses its own soundscapes.

- `console`: live bar with metric, threshold and reward status.
- `audio`: a reward tone whose volume follows how far past threshold you are. It's silent when
  not rewarded, so you can train with eyes closed. Needs `sounddevice`.
- `sonify`: hear your EEG as it happens. Each rhythm plays its own note from a pentatonic
  scale, so any mix sounds consonant. Slower rhythms are lower notes, placed across the
  stereo field: delta, theta, alpha (center), SMR, beta. High beta, often muscle tension,
  is a soft hiss. Each band's loudness follows its power compared with its own recent
  average, so you hear changes rather than raw levels. Needs `sounddevice`.
- `dashboard`: streams the session into the NOVA console in your browser, with soundscapes.
- `visual`: a pygame window with a circle that grows and glows with reward. Press Esc
  to stop. Needs `pygame`.

Combine them, for example `--feedback console,audio` or `--feedback sonify`.

## Auditory stimulation: binaural beats and 40 Hz tones

You can play a background sound during a session. NOVA measures whether your EEG
actually responds to it.

| `--stim` | Sound | Default |
|---|---|---|
| `binaural` | A slightly different tone in each ear. The brain perceives a "beat" at the difference. **Needs headphones.** | 10 Hz beat on a 200 Hz carrier |
| `pulsed` | One tone pulsing in volume (amplitude-modulated) | 40 Hz pulses on a 500 Hz carrier |

```bash
# Try the sound alone. No headband needed.
nova stim --stim binaural --stim-freq 6 --seconds 600
nova stim --stim pulsed --seconds 300 --out gamma40.wav   # save a file to play on a phone

# During a session: alternates 30 s on / 30 s off by default
nova run --device muse_2 --protocol alpha --stim pulsed --feedback console
```

By default, the stimulus alternates in blocks during training (`--stim-schedule blocks`).
At the end you get a direct comparison of the training metric and of EEG power at the
stimulation frequency, with the sound on and off. Each update in the CSV log records
`stim_on` and `stim_power`. Use `--stim-schedule training` or `always` to play it
continuously, and `--stim-volume` to adjust loudness (default 0.15; start low).

What to expect from the evidence:
- **40 Hz pulsed tones** reliably produce a measurable EEG response at 40 Hz in auditory
  cortex (the "auditory steady-state response"). Temporal sites pick it up best, such as
  Muse's TP9/TP10. When the stimulus is above 35 Hz, the analysis band is extended
  automatically so the response is visible. Jaw clenching also produces 40 Hz muscle
  noise, so keep your face relaxed.
- **Binaural beats:** claims that they shift your brain into a target rhythm have weak and
  inconsistent support in research. The on/off blocks let you test whether they do
  anything for you.

### Using AirPods or other Bluetooth headphones

Connect them as the computer's audio output; all sound goes through the default output
device. Tips:
- **Turn off Spatial Audio and head tracking.** They remix the stereo channels, which can
  wash out binaural beats.
- **Bluetooth adds about 150-250 ms of delay.** That's fine for neurofeedback rewards and
  stimulation. It isn't suitable for experiments that need millisecond sound timing.
- AirPods don't record EEG, so you still need a headband or other EEG device.

## Will my headband work?

Run `nova devices` and look for your model. Anything listed works with
`--device <name>`. If it isn't listed, check whether its app or a companion tool can
stream to **Lab Streaming Layer** (LSL); if so, use `--device lsl`. Many research
headsets can, and Muse can via `muselsl`.

Headbands that only output finished scores like "calm %" or "focus", with no raw EEG,
can't be used. Neurofeedback here needs the raw signal. Once connected, run
`nova check --device <name>` to confirm each channel shows `ok`.

## Using it as a library

```python
from nova import BrainFlowDevice, Session, SessionConfig
from nova.feedback import ConsoleFeedback
from nova.programs import get_program

device = BrainFlowDevice("muse_2")
program = get_program("calm-anxiety").scaled(0.6)
summary = Session(device, config=SessionConfig(), program=program,
                  sinks=[ConsoleFeedback()], log_path="my_session.csv").run()
print(summary.format())
```

To add your own hardware, subclass `nova.devices.Device` and implement `start`, `stop`
and `read`. `read` returns new samples in µV as a `(channels, samples)` array.

## Development

```bash
pip install -e ".[dev,brainflow]"
pytest
python -m nova.programs    # after changing built-in programs: refresh the app's embedded copy
```

Session time comes from the count of received samples, not the wall clock. So the test
suite runs every built-in program against the demo signal in fast mode in a few seconds.
