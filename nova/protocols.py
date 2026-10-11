"""Neurofeedback training protocols.

A protocol defines the metric being trained (one band's relative power, or a
ratio of two bands), which way it should move, and "inhibit" bands whose power
spikes (eye movement, muscle tension) withhold the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import band_power

Band = tuple[float, float]

BANDS: dict[str, Band] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "smr": (12.0, 15.0),
    "beta": (15.0, 22.0),
    "hibeta": (22.0, 30.0),
    "total": (1.0, 40.0),
}


@dataclass(frozen=True)
class Protocol:
    name: str
    description: str
    reward: str
    # When set, the metric is power(reward) / power(ratio_over); otherwise it is
    # reward-band power relative to total 1-40 Hz power.
    ratio_over: str | None = None
    # +1: reward when the metric goes up; -1: reward when it goes down.
    direction: int = 1
    inhibits: tuple[str, ...] = ()
    suggested_sites: tuple[str, ...] = field(default=())

    def metric(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        num = band_power(freqs, psd, BANDS[self.reward]).mean()
        den_band = BANDS[self.ratio_over] if self.ratio_over else BANDS["total"]
        den = band_power(freqs, psd, den_band).mean()
        return float(num / den) if den > 0 else 0.0

    def inhibit_powers(self, freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
        return {name: float(band_power(freqs, psd, BANDS[name]).mean()) for name in self.inhibits}

    @property
    def metric_label(self) -> str:
        if self.ratio_over:
            return f"{self.reward}/{self.ratio_over}"
        return f"rel. {self.reward}"


PROTOCOLS: dict[str, Protocol] = {
    p.name: p
    for p in (
        Protocol(
            name="alpha",
            description="Alpha up-training for relaxed, calm focus (eyes closed or soft gaze).",
            reward="alpha",
            inhibits=("delta", "hibeta"),
            suggested_sites=("Pz", "O1", "O2"),
        ),
        Protocol(
            name="smr",
            description="Sensorimotor rhythm (12-15 Hz) up-training for calm, still attention.",
            reward="smr",
            inhibits=("theta", "hibeta"),
            suggested_sites=("C4", "Cz"),
        ),
        Protocol(
            name="theta-beta",
            description="Lower the theta/beta ratio, a classic attention-training protocol.",
            reward="theta",
            ratio_over="beta",
            direction=-1,
            inhibits=("delta", "hibeta"),
            suggested_sites=("Cz", "Fz"),
        ),
        Protocol(
            name="alpha-theta",
            description="Raise theta relative to alpha for deep relaxation (eyes closed).",
            reward="theta",
            ratio_over="alpha",
            inhibits=("delta", "hibeta"),
            suggested_sites=("Pz", "O1", "O2"),
        ),
        Protocol(
            name="beta",
            description="Beta (15-22 Hz) up-training for alertness and engagement.",
            reward="beta",
            inhibits=("theta", "hibeta"),
            suggested_sites=("C3", "Fz"),
        ),
    )
}


def get_protocol(name: str) -> Protocol:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"unknown protocol {name!r}; choose from {', '.join(PROTOCOLS)}") from None
