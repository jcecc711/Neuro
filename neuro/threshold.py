"""Adaptive thresholds that keep the reward rate near a target.

Auto-thresholding is the standard way to keep neurofeedback challenging but
achievable: the threshold is re-estimated from the recent history of the
metric so that it is exceeded a chosen fraction of the time.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class AdaptiveThreshold:
    def __init__(self, target_rate: float, direction: int = 1, history: int = 120):
        if not 0.0 < target_rate < 1.0:
            raise ValueError("target_rate must be between 0 and 1")
        self.target_rate = target_rate
        self.direction = direction
        self._history: deque[float] = deque(maxlen=history)
        self.frozen = False
        self._value: float | None = None

    def observe(self, x: float) -> None:
        if self.frozen or not np.isfinite(x):
            return
        self._history.append(x)
        q = 1.0 - self.target_rate if self.direction > 0 else self.target_rate
        self._value = float(np.quantile(self._history, q))

    def freeze(self) -> None:
        self.frozen = True

    @property
    def ready(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> float:
        return self._value if self._value is not None else float("nan")

    @property
    def spread(self) -> float:
        return float(np.std(self._history)) if len(self._history) > 1 else 0.0

    def passes(self, x: float) -> bool:
        if not self.ready:
            return False
        return x > self._value if self.direction > 0 else x < self._value

    def intensity(self, x: float) -> float:
        """0..1 feedback strength: 0.5 at threshold, saturating ~1 SD beyond it."""
        if not self.ready:
            return 0.0
        scale = self.spread or abs(self._value) or 1.0
        return float(np.clip(0.5 + self.direction * (x - self._value) / (2 * scale), 0.0, 1.0))


class InhibitThreshold(AdaptiveThreshold):
    """Triggers when a band's power is unusually high (top `trigger_rate` of history)."""

    def __init__(self, trigger_rate: float = 0.1, history: int = 120):
        super().__init__(target_rate=trigger_rate, direction=1, history=history)

    def triggered(self, x: float) -> bool:
        return self.passes(x)
