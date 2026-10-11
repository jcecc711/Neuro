# Neuro: real-time EEG neurofeedback

Neuro reads EEG from your hardware, measures a brain rhythm in real time, and gives you
audio, visual or terminal feedback when it moves the way you're training it to.

> **Not a medical device.** This is for research, education and personal experimentation.
> It does not diagnose or treat any condition.

## Supported hardware

| Source | How | Examples |
|---|---|---|
| BrainFlow | `--device <board>` | OpenBCI Cyton / Ganglion / Daisy, Muse 2 / S, Neurosity Crown, BrainBit, g.tec Unicorn, and others |
| Lab Streaming Layer | `--device lsl` | Anything that publishes an LSL EEG stream (OpenBCI GUI, Muse via muselsl, BrainVision, ...) |
| Simulator | `--device sim` (default) | No hardware. Generates synthetic EEG with alpha that rises and falls |

Run `neuro devices` to list every BrainFlow board name.

## Install

```bash
pip install -e ".[all]"        # BrainFlow, LSL, audio and visual feedback
# or only what you need:
pip install -e ".[brainflow,audio]"
```

## Quick start

```bash
neuro protocols                                   # see training protocols
neuro run                                         # try it with the simulator

# OpenBCI Cyton over the USB dongle (use COM3 etc. on Windows)
neuro check --device cyton --serial-port /dev/ttyUSB0          # check electrode contact first
neuro run   --device cyton --serial-port /dev/ttyUSB0 --protocol smr --channels C4 \
            --feedback console,audio

# Muse 2 over native Bluetooth
neuro run --device muse_2 --protocol alpha --channels TP9,TP10 --feedback visual

# Any LSL stream
neuro run --device lsl --lsl-name "OpenBCI_EEG" --protocol theta-beta
```

Use `--line-freq 50` in Europe, Asia and other regions with 50 Hz mains power.

## How a session works

1. **Baseline** (`--baseline`, default 60 s). Your normal levels are measured. No rewards
   are given.
2. **Training** (`--duration`, default 300 s). Every 250 ms the last 2 s of EEG is
   band-pass filtered (1-40 Hz) with a mains notch, and band powers are computed. You are rewarded when:
   - the protocol metric is past an **adaptive threshold**, set so you hit it about
     `--reward-rate` of the time (default 60%),
   - no **inhibit band** (for example, theta from eye movement or high beta from muscle
     tension) is spiking,
   - the signal has no **artifact** (peak-to-peak above `--artifact-uv`),
   - and all of that has held for `--sustain` seconds.
3. **Summary and log.** You get a summary of baseline vs. training metric and time
   rewarded. A per-update CSV is written to `sessions/`.

Thresholds keep adapting during training. Use `--freeze-thresholds` to fix them at the
end of baseline instead.

## Protocols

| Name | Trains | Typical sites |
|---|---|---|
| `alpha` | relative alpha (8-12 Hz) up | Pz, O1, O2 |
| `smr` | relative SMR (12-15 Hz) up | C4, Cz |
| `theta-beta` | theta/beta ratio down | Cz, Fz |
| `alpha-theta` | theta/alpha ratio up | Pz, O1, O2 |
| `beta` | relative beta (15-22 Hz) up | C3, Fz |

To add a protocol, add a `Protocol(...)` entry in `neuro/protocols.py`.

## Feedback

- `console`: live bar with metric, threshold and reward status.
- `audio`: a reward tone whose volume follows how far past threshold you are. It's silent when
  not rewarded, so you can train with eyes closed. Needs `sounddevice`.
- `sonify`: hear your EEG as it happens. Each rhythm plays its own note from a pentatonic
  scale, so any mix sounds consonant. Slower rhythms are lower notes, placed across the
  stereo field: delta, theta, alpha (center), SMR, beta. High beta, often muscle tension,
  is a soft hiss. Each band's loudness follows its power compared with its own recent
  average, so you hear changes rather than raw levels. Needs `sounddevice`.
- `visual`: a pygame window with a circle that grows and glows with reward. Press Esc
  to stop. Needs `pygame`.

Combine them, for example `--feedback console,audio` or `--feedback sonify`.

## Auditory stimulation: binaural beats and 40 Hz tones

You can play a background sound during a session. Neuro measures whether your EEG
actually responds to it.

| `--stim` | Sound | Default |
|---|---|---|
| `binaural` | A slightly different tone in each ear. The brain perceives a "beat" at the difference. **Needs headphones.** | 10 Hz beat on a 200 Hz carrier |
| `pulsed` | One tone pulsing in volume (amplitude-modulated) | 40 Hz pulses on a 500 Hz carrier |

```bash
# Try the sound alone. No headband needed.
neuro stim --stim binaural --stim-freq 6 --seconds 600
neuro stim --stim pulsed --seconds 300 --out gamma40.wav   # save a file to play on a phone

# During a session: alternates 30 s on / 30 s off by default
neuro run --device muse_2 --protocol alpha --stim pulsed --feedback console
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

Run `neuro devices` and look for your model. Anything listed works with
`--device <name>`. If it isn't listed, check whether its app or a companion tool can
stream to **Lab Streaming Layer** (LSL); if so, use `--device lsl`. Many research
headsets can, and Muse can via `muselsl`.

Headbands that only output finished scores like "calm %" or "focus", with no raw EEG,
can't be used. Neurofeedback here needs the raw signal. Once connected, run
`neuro check --device <name>` to confirm each channel shows `ok`.

## Using it as a library

```python
from neuro import BrainFlowDevice, Session, SessionConfig, get_protocol
from neuro.feedback import ConsoleFeedback

device = BrainFlowDevice("ganglion", serial_port="/dev/ttyACM0")
cfg = SessionConfig(baseline_sec=60, training_sec=600, channels=["EEG1"])
summary = Session(device, get_protocol("smr"), cfg, [ConsoleFeedback()], "my_session.csv").run()
print(summary.format())
```

To add your own hardware, subclass `neuro.devices.Device` and implement `start`, `stop`
and `read`. `read` returns new samples in µV as a `(channels, samples)` array.

## Development

```bash
pip install -e ".[dev,brainflow]"
pytest
```

Session time comes from the count of received samples, not the wall clock. So the test
suite runs full sessions against the simulator in fast mode in a few seconds.
