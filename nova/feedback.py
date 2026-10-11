"""Feedback outputs: terminal bar, audio tone and a full-screen visual."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from .audio import AudioEngine, NoiseVoice, ToneVoice, Voice


class SessionAborted(Exception):
    """Raised by a feedback sink when the user closes it."""


@dataclass
class FeedbackState:
    t: float
    phase: str  # "baseline" or "training"
    metric: float
    threshold: float
    reward: bool
    intensity: float  # 0..1
    inhibited: list[str] = field(default_factory=list)
    artifact: bool = False
    reward_fraction: float = 0.0
    band_powers: dict[str, float] = field(default_factory=dict)  # relative to 1-40 Hz
    stim_on: bool = False
    samples: np.ndarray | None = None  # filtered µV received since the previous update


class FeedbackSink:
    def configure(self, session) -> None:
        """Called with the running Session before open(), for sinks that need its setup."""

    def open(self) -> None:
        pass

    def update(self, state: FeedbackState) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ConsoleFeedback(FeedbackSink):
    def __init__(self, stream=None, width: int = 30):
        self.stream = stream or sys.stdout
        self.width = width

    def update(self, s: FeedbackState) -> None:
        filled = int(round(s.intensity * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        if s.phase == "baseline":
            status = "calibrating"
        elif s.artifact:
            status = "ARTIFACT"
        elif s.inhibited:
            status = "inhibit:" + ",".join(s.inhibited)
        else:
            status = "REWARD" if s.reward else ""
        if s.stim_on:
            status = ("~ " + status).strip()
        line = (
            f"\r{s.t:6.1f}s {s.phase:<8} [{bar}] metric={s.metric:.3f} "
            f"thr={s.threshold:.3f} rewarded={s.reward_fraction:5.1%} {status:<16}"
        )
        self.stream.write(line)
        self.stream.flush()

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()


class AudioFeedback(FeedbackSink):
    """Continuous tone whose volume follows feedback intensity; silent unless rewarded."""

    def __init__(self, engine: AudioEngine, frequency: float = 440.0, volume: float = 0.3):
        self.engine = engine
        self.volume = volume
        self.voice = ToneVoice(frequency, harmonics=(0.3, 0.1))

    def open(self) -> None:
        self.engine.add(self.voice)
        self.engine.start()

    def update(self, s: FeedbackState) -> None:
        rewarded = s.reward and s.phase == "training"
        self.voice.target = self.volume * s.intensity if rewarded else 0.0

    def close(self) -> None:
        self.engine.remove(self.voice)
        self.engine.stop()


# Each band gets a note from the A minor pentatonic scale so any mix sounds consonant;
# slower rhythms are lower notes. High beta (often muscle tension) is a soft hiss.
SONIFY_VOICES: dict[str, tuple[float, float]] = {
    "delta": (110.0, -0.6),
    "theta": (164.8, -0.3),
    "alpha": (220.0, 0.0),
    "smr": (329.6, 0.3),
    "beta": (440.0, 0.6),
}


class SonifyFeedback(FeedbackSink):
    """Hear your EEG: each frequency band drives its own tone.

    A band's loudness tracks its relative power compared with its own running
    average, so you hear changes ("alpha is rising") rather than raw levels,
    which are dominated by slow waves.
    """

    def __init__(self, engine: AudioEngine, volume: float = 0.35, adapt_sec: float = 30.0, update_sec: float = 0.25):
        self.engine = engine
        self.volume = volume
        self._alpha = min(1.0, update_sec / adapt_sec)
        self.voices: dict[str, Voice] = {
            band: ToneVoice(freq, pan=pan, harmonics=(0.25,)) for band, (freq, pan) in SONIFY_VOICES.items()
        }
        self.voices["hibeta"] = NoiseVoice(brightness=0.25)
        self._means: dict[str, float] = {}

    def open(self) -> None:
        for v in self.voices.values():
            self.engine.add(v)
        self.engine.start()

    def levels(self, band_powers: dict[str, float]) -> dict[str, float]:
        """Update running means and return a 0..1 loudness per band."""
        out = {}
        for band in self.voices:
            p = band_powers.get(band)
            if p is None or not np.isfinite(p):
                out[band] = 0.0
                continue
            mean = self._means.get(band, p)
            mean += self._alpha * (p - mean)
            self._means[band] = mean
            ratio = p / mean if mean > 0 else 1.0
            # ratio 1 (typical) -> 0.33; 2x typical -> ~1; below half typical -> silent.
            out[band] = float(np.clip((ratio - 0.5) / 1.5, 0.0, 1.0))
        return out

    def update(self, s: FeedbackState) -> None:
        levels = self.levels(s.band_powers)
        gain = 0.0 if s.artifact else self.volume / len(self.voices) * 2
        for band, voice in self.voices.items():
            voice.target = gain * levels[band]

    def close(self) -> None:
        for v in self.voices.values():
            self.engine.remove(v)
        self.engine.stop()


class VisualFeedback(FeedbackSink):
    """Pygame window: a circle that grows and brightens as the brain hits the target."""

    def __init__(self, size: tuple[int, int] = (800, 600), fullscreen: bool = False):
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError("visual feedback needs pygame: pip install 'nova[visual]'") from e
        self._pg = pygame
        self.size = size
        self.fullscreen = fullscreen
        self._screen = None
        self._font = None
        self._shown = 0.0

    def open(self) -> None:
        pg = self._pg
        pg.init()
        flags = pg.FULLSCREEN if self.fullscreen else 0
        self._screen = pg.display.set_mode((0, 0) if self.fullscreen else self.size, flags)
        pg.display.set_caption("Neurofeedback")
        self._font = pg.font.SysFont(None, 28)

    def update(self, s: FeedbackState) -> None:
        pg = self._pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                raise SessionAborted()

        target = s.intensity if s.phase == "training" and not s.artifact else 0.0
        self._shown += 0.3 * (target - self._shown)  # smooth to avoid flicker

        w, h = self._screen.get_size()
        self._screen.fill((8, 10, 20))
        radius = int(min(w, h) * (0.08 + 0.32 * self._shown))
        glow = int(60 + 195 * self._shown)
        color = (glow // 3, glow, int(glow * 0.8)) if s.reward else (60, 70, 110)
        pg.draw.circle(self._screen, color, (w // 2, h // 2), radius)

        if s.phase == "baseline":
            caption = "Calibrating - relax and stay still"
        else:
            caption = f"Rewarded {s.reward_fraction:.0%}"
            if s.artifact:
                caption += "   (movement detected)"
        text = self._font.render(caption, True, (200, 200, 210))
        self._screen.blit(text, (20, h - 40))
        pg.display.flip()

    def close(self) -> None:
        if self._screen is not None:
            self._pg.quit()
            self._screen = None


FEEDBACK_KINDS = ("console", "audio", "sonify", "visual", "dashboard")


def make_sinks(
    names: list[str], engine: AudioEngine | None = None, dashboard_port: int = 8765
) -> list[FeedbackSink]:
    from .live import DashboardFeedback

    engine = engine or AudioEngine()
    factories = {
        "console": ConsoleFeedback,
        "audio": lambda: AudioFeedback(engine),
        "sonify": lambda: SonifyFeedback(engine),
        "visual": VisualFeedback,
        "dashboard": lambda: DashboardFeedback(port=dashboard_port),
    }
    sinks = []
    for name in names:
        if name not in factories:
            raise ValueError(f"unknown feedback {name!r}; choose from {', '.join(FEEDBACK_KINDS)}")
        sinks.append(factories[name]())
    return sinks
