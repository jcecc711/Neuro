"""Background auditory stimulation (binaural beats, pulsed tones) with on/off scheduling.

Alternating blocks of stimulation on and off during training lets a session
compare EEG with and without the sound, which is the only way to tell whether
it actually did anything for you.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio import AudioEngine, BinauralVoice, PulsedToneVoice, Voice

SCHEDULES = ("always", "training", "blocks")


@dataclass(frozen=True)
class StimulusSpec:
    kind: str  # "binaural" or "pulsed"
    freq: float  # beat / modulation frequency in Hz
    carrier: float
    volume: float = 0.15
    schedule: str = "training"
    block_sec: float = 30.0

    @classmethod
    def create(
        cls,
        kind: str,
        freq: float | None = None,
        carrier: float | None = None,
        volume: float = 0.15,
        schedule: str = "training",
        block_sec: float = 30.0,
    ) -> "StimulusSpec":
        defaults = {"binaural": (10.0, 200.0), "pulsed": (40.0, 500.0)}
        if kind not in defaults:
            raise ValueError(f"unknown stimulus {kind!r}; choose from {', '.join(defaults)}")
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}; choose from {', '.join(SCHEDULES)}")
        if not 0.0 < volume <= 1.0:
            raise ValueError("stimulus volume must be in (0, 1]")
        d_freq, d_carrier = defaults[kind]
        return cls(kind, freq or d_freq, carrier or d_carrier, volume, schedule, block_sec)

    @property
    def sound(self) -> str:
        if self.kind == "binaural":
            return f"binaural beat {self.freq:g} Hz (carrier {self.carrier:g} Hz)"
        return f"{self.freq:g} Hz pulsed tone (carrier {self.carrier:g} Hz)"

    @property
    def description(self) -> str:
        when = {
            "always": "on for the whole session",
            "training": "on during training",
            "blocks": f"alternating {self.block_sec:g}s on / {self.block_sec:g}s off during training",
        }[self.schedule]
        return f"{self.sound}, {when}"

    def make_voice(self) -> Voice:
        if self.kind == "binaural":
            return BinauralVoice(self.carrier, self.freq)
        return PulsedToneVoice(self.carrier, self.freq)

    def active_at(self, t: float, baseline_sec: float) -> bool:
        if self.schedule == "always":
            return True
        if t < baseline_sec:
            return False
        if self.schedule == "training":
            return True
        return int((t - baseline_sec) // self.block_sec) % 2 == 0


class Stimulus:
    """Plays a StimulusSpec through an AudioEngine and gates it on schedule."""

    def __init__(self, spec: StimulusSpec, engine: AudioEngine):
        self.spec = spec
        self.engine = engine
        self.voice = spec.make_voice()
        self.on = False

    def open(self) -> None:
        self.engine.add(self.voice)
        self.engine.start()

    def step(self, t: float, baseline_sec: float) -> bool:
        self.on = self.spec.active_at(t, baseline_sec)
        self.voice.target = self.spec.volume if self.on else 0.0
        return self.on

    def close(self) -> None:
        self.voice.target = 0.0
        self.engine.remove(self.voice)
        self.engine.stop()
