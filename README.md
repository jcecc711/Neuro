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
- `audio`: a tone whose volume follows how far past threshold you are. It's silent when
  not rewarded, so you can train with eyes closed. Needs `sounddevice`.
- `visual`: a pygame window with a circle that grows and glows with reward. Press Esc
  to stop. Needs `pygame`.

Combine them, for example `--feedback console,audio`.

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
