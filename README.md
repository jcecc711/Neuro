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

Without a headset, choose **Demo signal** to try any program with generated EEG and the full
soundscape. In demo sessions the trained rhythm swells and fades at the program's target sites,
so you can see and hear the feedback respond.

**Session length:** 15, 20, 30, 45 or 60 minutes. Arrival, warm-up and integration keep their
length; the rest becomes training blocks of about 5-8 minutes with one-minute breathing
interludes, cycling through the program's blocks (`nova run --program focus-adhd --minutes 45`).

### iPad and AirPods

NOVA runs in Safari on iPad. Connect AirPods to the iPad and turn off Spatial Audio. Sessions
keep the screen awake and play even with the silent switch on. Browsers on iPad can't connect
to EEG headsets, so to train with a headset run the engine on a computer and open it from the
iPad over Wi-Fi:

```bash
nova app --host 0.0.0.0     # prints an address like http://192.168.1.20:8765 for the iPad
```

Anyone on that network can open it, so use it on a home network you trust.

**Full screen in landscape.** With the iPad (or a wide window) in landscape, the session fits
one screen with no scrolling: the brain fills the left, the on-target ring and live activity sit
on the right, the band cards and timeline run along the bottom, and **Details** slides up the
phase list, spectrum and other tabs. For an edge-to-edge app without Safari's bars, open
`nova app` in Safari, tap Share, then **Add to Home Screen**; NOVA then launches full screen in
landscape from its own icon. In Safari you can also tap the full-screen button in the brain viewer
where the browser allows it, or hide the toolbar from the **aA** menu.

**Hosted app (no computer needed for demo sessions).** `python -m nova.site` builds NOVA as a
static web app in `_site/` (page, manifest, icons and an offline service worker), and the
`Pages` GitHub Actions workflow publishes it to GitHub Pages on every push. Open the Pages
address in Safari on the iPad, tap Share > **Add to Home Screen**, and NOVA opens full screen in
landscape from its icon, even offline. To turn it on: repository **Settings > Pages > Source:
GitHub Actions** (Pages on a private repository needs a paid GitHub plan).

**On a Mac.** Open the same address in Safari and choose **File > Add to Dock** (macOS Sonoma or
later): NOVA gets its own Dock icon and window; press the green window button or Ctrl-Cmd-F for
full screen. On large displays the session spreads across the full width. Use headphones for the
3D sound and the rhythm cue.

### Learn and the training plan

The **Learn** section explains what's different in the ADHD brain, what EEG shows (and doesn't),
how each protocol targets it, and how training can help impulses pass. It also covers the systems
behind willpower and motivation. Its two-track plan counts your sessions:

1. **Attention and impulse control:** Focus Builder and Impulse Control, 20 sessions.
2. **Willpower, discipline and positive action:** Willpower (frontal-midline theta) and Positive
   Action (left-frontal approach), 20 sessions.

### What's on screen during a session

- **3D brain** with anatomical lobes, fissures and folds, coloured by live activity at each
  sensor. Drag or swipe to rotate (it glides with momentum), pinch to zoom on iPad or a
  trackpad, use the scroll wheel or the +/- buttons on desktop, and double-tap to reset.
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
| Impulse Control | ADHD impulsivity, restlessness | SMR with strict inhibits, theta/beta | C4, Cz | Most studied |
| Willpower | Self-control, discipline | frontal-midline theta, SMR | Fz | Emerging |
| Positive Action | Motivation, follow-through | frontal asymmetry, left-frontal beta | F3/F4 | Emerging |

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
- the soundscape, eyes open or closed, and the rhythm cue frequency (it matches the target band unless you pick one).

Programs are saved to `~/.nova/programs/` as JSON, which you can edit or share.

## Soundscapes

Every session plays like a gentle, uplifting pop song, synthesized live in the browser and built
from the traits that recur across popular songs: a major key, a classic four-chord loop, a steady
4/4 pulse at 66-84 BPM, and a short vocal hook that repeats. It's arranged as a **3D sound bath**:
each instrument sits at its own place around your head (HRTF panning), and the backing vocals drift
in slow orbits.

- **Ensemble:** flowing harp arpeggios (front left), a lush string section spread wide on both
  sides, warm synth pads and a high shimmer above, a round bass, and only a soft heartbeat kick
  and brushes for rhythm. Every note is humanized in timing and touch, so it never sounds
  mechanical.
- **Voices:** wordless female vocals ("ooh", "oh", "ah") sing a four-bar hook that repeats, with a
  harmony a third below and stacked "ooh" chords in the chorus. Switch them off with **Voices**.
- **Song form follows the session:** verse while NOVA calibrates (harp, strings, bass), a harp
  glissando into the chorus as training starts, a stripped-back bridge for breathing breaks,
  and an outro through the cooldown.
- **Feedback:** being on target opens up the chorus (upper strings, heartbeat and brushes, harmonies, shimmer); drifting off
  settles it back to the verse. A reward after a quiet stretch plays a rising harp glissando.
- **Targeting:** during training the lead vocal and the rhythm cue move to the brain region being
  trained: right of the crown for SMR at C4, the crown for theta/beta at Cz, behind for alpha at
  Pz, front-left for asymmetry at F3, and front-centre for frontal theta at Fz.

| Soundscape | Character |
|---|---|
| Aurora | 76 BPM, I-V-vi-IV in D, harp, strings and an "ooh" vocal hook |
| Deep Ocean | 68 BPM, vi-IV-I-V in C, gentle waves under harp and warm pads |
| Celestial | 82 BPM, I-vi-IV-V in E, soaring strings and stacked harmonies |
| Forest Rain | 78 BPM, IV-I-V-vi in G, soft rain, harp and brushes |
| Nebula | 66 BPM, I-iii-vi-IV in F, slow, dreamy synth pads |
| Dawn | 84 BPM, the Canon progression in A, uplifting strings and harp |

**Recorded instruments (Surge XT).** The harp, strings, warm pad and voices are real
renders from [Surge XT](https://surge-synthesizer.github.io/), the free, open-source synthesizer
(GPL-3.0): "Orchestral Harp For Runs" and "Violin Section" by John Valentine, and the factory "MKS-70 Warm Pad", "Synth Choir MW O-Ah" and "Ooh". One note every
three semitones is stored in `nova/dashboard/samples/` (about 3.5 MB of MP3); NOVA pitches each to
the exact frequency it needs, loads notes the first time they're used, and plays its built-in
synthesis until they arrive (or when offline before they've been cached). Strings and pads play legato:
a voice keeps bowing through a chord change when its note stays the same, and otherwise the old
note fades over a few seconds while the new one swells in. The harp's recorded pluck rings into a
soft, long-sustaining body so its arpeggios flow together.

**Light sync (optional, off by default).** The **☼ Light sync** button in the session view (also under
**Details**) adds full-screen light
in time with the entrainment rhythm, during training only:

- **40 Hz (gamma, e.g. Clear Mind):** a hard on/off square wave, as in MIT's GENUS research, locked
  to a GENUS-style 1 kHz click train (switched on automatically with the light) using the audio
  clock. NOVA measures the display's refresh rate and runs it only when 40 Hz can be shown exactly:
  3 frames per cycle on a 120 Hz display (33% duty), 6 on a 240 Hz display (50% duty). A 60 Hz
  screen would produce an irregular flicker with a 20 Hz component, so NOVA refuses and says why.
  On iPad Pro, Safari may cap pages at 60 fps unless "Prefer Page Rendering Updates near 60fps" is
  turned off in Settings > Apps > Safari > Advanced > Feature Flags.
- **Slower targets (alpha, SMR, beta, theta):** 3-30 Hz flicker is the range most likely to trigger
  photosensitive seizures, so these get a soft warm glow that pulses at the rhythm by under 4%,
  below the standard 10% flash threshold. It's gentle company for the sound, not strong entrainment.
- **Safety:** turning it on shows a seizure and migraine warning that must be accepted; a **Stop
  light** button stays on screen, and Esc, leaving the app or ending the session stops it at once.
  It's disabled when the device asks for reduced motion.
- **Limits:** a screen is far dimmer than the clinical LED panels used in trials (hundreds of lux at
  eye level), and trials used 60-minute daily sessions over months; NOVA's light is a wellness
  feature, not a medical treatment. Pick the 60-minute length for the closest match.

The rhythm cue's **Click train** style can also be chosen on its own.

**Production touches.** A second, 14-second shimmer reverb turns the harp, pads and octave
shimmer into an endless wash, the long-tail sound of ambient production. Under **Details** you can
choose the **tuning** (standard A = 440 Hz, A = 432 Hz, or Solfeggio 528 Hz, which nudges the key so
a scale note lands exactly on 528 Hz) and the **rhythm cue** style: binaural beats with a gentle
pulse (headphones), or isochronic pulses, a soft-edged tone switching on and off at the exact rate,
which work on any speaker. There is no scientific evidence that 432 or 528 Hz tuning affects the
brain; it's a matter of taste, and the entrainment frequencies are exact either way.

Use stereo headphones for the 3D effect. With AirPods, turn **Spatial Audio off** (Control Centre,
press and hold the volume slider): NOVA does its own 3D placement, and Apple's would remix it.

### Adaptive entrainment

NOVA steers the music toward the state being trained, closed-loop, from the live EEG:

- **Pace and lead.** Each training block starts the rhythm at your brain's current dominant
  frequency near the target (within 4 Hz), then glides it to the target, only as fast as your brain
  follows (measured as time on target over the last 15 seconds). If you lose it for a minute, it
  meets your brain where it is and leads again.
- **Modulated music.** The whole song pulses gently in loudness at that frequency (amplitude
  modulation), alongside the rhythm cue below. The depth grows slowly while your brain resists and
  relaxes once it's there, capped at 30% for comfort.
- **Tempo and energy suit the goal.** Calming targets (alpha, theta) slow the song by up to 6% and
  keep the drums light; focus targets (SMR, beta) lift it slightly with the full groove.

**40 Hz (Clear Mind).** The Clear Mind program pairs alertness training with the soundtrack and
rhythm cue pulsing at exactly 40 Hz during training, the gamma rhythm used in MIT's GENUS sensory
stimulation research. That work is mostly in mouse models of Alzheimer's with small early human
trials; effects on everyday brain fog are anecdotal so far. Calibration is left unstimulated.

The readout shows the current rhythm and the target, e.g. "12.4 → 13.5 Hz". Switch all of it off
with **Entrainment**. Honest limits: auditory entrainment and modulated music have some supportive
studies but mixed evidence overall; they nudge rather than force a brain state. The feedback, being
rewarded when your own brain gets there, remains the core of the training.

### Rhythm cue: the exact target frequency

During training, a quiet **rhythm cue** pulses at the exact frequency of the protocol you're
training: a binaural pair whose ears differ by exactly that many Hz (the left ear on the
soundscape's key note at 110-220 Hz, the right ear exactly the cue frequency above it), plus a
soft tone whose loudness pulses at the same rate, placed at the trained region. It eases in over
each training block's first 45 seconds and bypasses the reverb so the rhythm stays exact.

| Protocol | Cue | Why |
|---|---|---|
| alpha, alpha asymmetry | 10 Hz | centre of alpha (8-12 Hz) |
| SMR, SMR-still | 13.5 Hz | centre of SMR (12-15 Hz) |
| theta/beta (down) | 16 Hz | the beta side, since the goal is more beta relative to theta |
| alpha-theta | 7.5 Hz | the theta-alpha crossover |
| beta, beta-left | 18 Hz | centre of beta (15-22 Hz) |
| frontal-midline theta | 6 Hz | centre of theta (4-8 Hz) |

It's off during the baseline so calibration hears only your brain, quieter during rest, and fades
out across the cooldown. Binaural beats need headphones (AirPods work; turn Spatial Audio off).
The cue is complementary: the training itself is the feedback, and evidence that audio rhythms
entrain the brain is weak (see below). Turn it off with **Rhythm cue**.

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
