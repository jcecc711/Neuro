"""Feedback outputs: terminal bar, audio tone and a full-screen visual."""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass, field

import numpy as np


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


class FeedbackSink:
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

    def __init__(self, frequency: float = 440.0, volume: float = 0.3, sample_rate: int = 44100):
        try:
            import sounddevice
        except ImportError as e:
            raise RuntimeError("audio feedback needs sounddevice: pip install 'neuro[audio]'") from e
        self._sd = sounddevice
        self.frequency = frequency
        self.volume = volume
        self.sample_rate = sample_rate
        self._phase = 0.0
        self._amp = 0.0
        self._target = 0.0
        self._lock = threading.Lock()
        self._stream = None

    def open(self) -> None:
        self._stream = self._sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._callback
        )
        self._stream.start()

    def _callback(self, outdata, frames, _time, _status) -> None:
        with self._lock:
            target = self._target
        # Ramp amplitude across the block to avoid clicks.
        amps = np.linspace(self._amp, target, frames, dtype=np.float32)
        self._amp = target
        step = 2 * math.pi * self.frequency / self.sample_rate
        phases = self._phase + step * np.arange(frames)
        self._phase = float((phases[-1] + step) % (2 * math.pi))
        outdata[:, 0] = amps * np.sin(phases).astype(np.float32)

    def update(self, s: FeedbackState) -> None:
        target = self.volume * s.intensity if (s.reward and s.phase == "training") else 0.0
        with self._lock:
            self._target = target

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class VisualFeedback(FeedbackSink):
    """Pygame window: a circle that grows and brightens as the brain hits the target."""

    def __init__(self, size: tuple[int, int] = (800, 600), fullscreen: bool = False):
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError("visual feedback needs pygame: pip install 'neuro[visual]'") from e
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


def make_sinks(names: list[str]) -> list[FeedbackSink]:
    factories = {"console": ConsoleFeedback, "audio": AudioFeedback, "visual": VisualFeedback}
    sinks = []
    for name in names:
        if name not in factories:
            raise ValueError(f"unknown feedback {name!r}; choose from {', '.join(factories)}")
        sinks.append(factories[name]())
    return sinks
