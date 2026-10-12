"""Neurofeedback training protocols.

A protocol defines the metric being trained, which way it should move, the
scalp sites it is measured at, and "inhibit" bands whose power spikes (eye
movement, muscle tension) withhold the reward. Metrics are:

- power:     reward-band power relative to total 1-40 Hz power
- ratio:     power(reward) / power(ratio_over)
- asymmetry: ln(right reward power) - ln(left reward power) at a homologous pair
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
    "gamma": (30.0, 40.0),
    "g40": (38.0, 42.0),  # the brain's response to 40 Hz stimulation (auditory/visual steady state)
    "total": (1.0, 40.0),
}

# Centre frequency of each band, used for the soundscape's rhythm cue.
CUE_HZ: dict[str, float] = {"delta": 2.5, "theta": 6.0, "alpha": 10.0, "smr": 13.5, "beta": 18.0, "hibeta": 25.0, "gamma": 35.0, "g40": 40.0}

# Homologous left/right pairs, best first, for asymmetry protocols.
FRONTAL_PAIRS: tuple[tuple[str, str], ...] = (("F3", "F4"), ("AF3", "AF4"), ("AF7", "AF8"), ("F7", "F8"), ("Fp1", "Fp2"))


@dataclass(frozen=True)
class Protocol:
    name: str
    description: str
    reward: str
    ratio_over: str | None = None
    # +1: reward when the metric goes up; -1: reward when it goes down.
    direction: int = 1
    inhibits: tuple[str, ...] = ()
    # Preferred sites. Those present on the headset are used; if none are, all channels are.
    sites: tuple[str, ...] = field(default=())
    kind: str = "power"  # "power", "ratio" or "asymmetry"
    pairs: tuple[tuple[str, str], ...] = ()
    cue: float | None = None  # rhythm-cue frequency in Hz; defaults from the target band

    def __post_init__(self) -> None:
        for band in (self.reward, self.ratio_over, *self.inhibits):
            if band is not None and band not in BANDS:
                raise ValueError(f"unknown band {band!r}; choose from {', '.join(b for b in BANDS if b != 'total')}")
        if self.direction not in (1, -1):
            raise ValueError("direction must be 1 (increase) or -1 (decrease)")
        if self.ratio_over and self.kind == "power":
            object.__setattr__(self, "kind", "ratio")

    @classmethod
    def from_spec(cls, spec: dict) -> "Protocol":
        """Build a custom protocol from a dict, e.g. from a saved program file."""
        reward = spec["reward"]
        ratio = spec.get("ratio_over") or None
        direction = 1 if spec.get("direction", "up") in ("up", 1, "+1", "increase") else -1
        name = spec.get("name") or f"{reward}{'/' + ratio if ratio else ''}-{'up' if direction > 0 else 'down'}"
        return cls(
            name=name,
            description=spec.get("description", "Custom protocol"),
            reward=reward,
            ratio_over=ratio,
            direction=direction,
            inhibits=tuple(spec.get("inhibits", ("delta", "hibeta"))),
            sites=tuple(spec.get("sites", ())),
        )

    @property
    def cue_hz(self) -> float:
        """The frequency the soundscape's rhythm cue pulses at: the centre of the rhythm the
        protocol is building (for a ratio trained down, the rhythm in the denominator)."""
        if self.cue is not None:
            return self.cue
        if self.kind == "asymmetry":
            return CUE_HZ["alpha"]
        if self.ratio_over and self.direction < 0:
            return CUE_HZ[self.ratio_over]
        return CUE_HZ[self.reward]

    def to_spec(self) -> dict:
        return {
            "cue_hz": self.cue_hz,
            "name": self.name,
            "reward": self.reward,
            "ratio_over": self.ratio_over,
            "direction": "up" if self.direction > 0 else "down",
            "inhibits": list(self.inhibits),
            "sites": list(self.sites),
            "kind": self.kind,
            "label": self.metric_label,
        }

    # Kept for older callers and the CLI listing.
    @property
    def suggested_sites(self) -> tuple[str, ...]:
        return self.pairs[0] if self.kind == "asymmetry" and self.pairs else self.sites

    def site_indices(self, channel_names: list[str] | None, n_channels: int) -> list[int]:
        if channel_names:
            idx = [channel_names.index(s) for s in self.sites if s in channel_names]
            if idx:
                return idx
        return list(range(n_channels))

    def pair_indices(self, channel_names: list[str] | None) -> tuple[int, int] | None:
        if not channel_names:
            return None
        for left, right in self.pairs:
            if left in channel_names and right in channel_names:
                return channel_names.index(left), channel_names.index(right)
        return None

    def check_channels(self, channel_names: list[str]) -> str | None:
        """Return a problem description if this protocol can't run on these channels."""
        if self.kind == "asymmetry" and self.pair_indices(channel_names) is None:
            wanted = ", ".join(f"{l}/{r}" for l, r in self.pairs)
            return f"protocol '{self.name}' needs a left/right channel pair ({wanted})"
        return None

    def metric(self, freqs: np.ndarray, psd: np.ndarray, channel_names: list[str] | None = None) -> float:
        if self.kind == "asymmetry":
            pair = self.pair_indices(channel_names)
            if pair is None:
                return float("nan")
            power = band_power(freqs, psd[list(pair)], BANDS[self.reward])
            left, right = np.maximum(power, 1e-12)
            return float(np.log(right) - np.log(left))
        rows = psd[self.site_indices(channel_names, psd.shape[0])]
        num = band_power(freqs, rows, BANDS[self.reward]).mean()
        den = band_power(freqs, rows, BANDS[self.ratio_over or "total"]).mean()
        return float(num / den) if den > 0 else 0.0

    def inhibit_powers(
        self, freqs: np.ndarray, psd: np.ndarray, channel_names: list[str] | None = None
    ) -> dict[str, float]:
        rows = psd[self.site_indices(channel_names, psd.shape[0])]
        return {name: float(band_power(freqs, rows, BANDS[name]).mean()) for name in self.inhibits}

    @property
    def metric_label(self) -> str:
        if self.kind == "asymmetry":
            return f"{self.reward} asymmetry (R-L)"
        if self.ratio_over:
            return f"{self.reward}/{self.ratio_over}"
        if self.reward == "g40":
            return "40 Hz response"
        return f"rel. {self.reward}"


PROTOCOLS: dict[str, Protocol] = {
    p.name: p
    for p in (
        Protocol(
            name="alpha",
            description="Alpha up-training for relaxed, calm focus (eyes closed or soft gaze).",
            reward="alpha",
            inhibits=("delta", "hibeta"),
            sites=("Pz", "O1", "O2", "P3", "P4", "TP9", "TP10"),
        ),
        Protocol(
            name="smr",
            description="Sensorimotor rhythm (12-15 Hz) up-training for calm, still attention.",
            reward="smr",
            inhibits=("theta", "hibeta"),
            sites=("C4", "Cz", "C3"),
        ),
        Protocol(
            name="theta-beta",
            description="Lower the theta/beta ratio, a classic attention-training protocol.",
            reward="theta",
            ratio_over="beta",
            direction=-1,
            inhibits=("delta", "hibeta"),
            sites=("Cz", "Fz", "FCz"),
            cue=16.0,  # low beta: alert but calm
        ),
        Protocol(
            name="alpha-theta",
            description="Raise theta relative to alpha for deep relaxation (eyes closed).",
            reward="theta",
            ratio_over="alpha",
            inhibits=("delta", "hibeta"),
            sites=("Pz", "O1", "O2", "TP9", "TP10"),
            cue=7.5,  # the alpha-theta crossover
        ),
        Protocol(
            name="beta",
            description="Beta (15-22 Hz) up-training for alertness and engagement.",
            reward="beta",
            inhibits=("theta", "hibeta"),
            sites=("Cz", "Fz", "C3"),
        ),
        Protocol(
            name="alpha-asymmetry",
            description="Shift frontal alpha toward the right (relatively more left-frontal activity), "
            "a protocol studied for low mood.",
            reward="alpha",
            inhibits=("delta", "hibeta"),
            kind="asymmetry",
            pairs=FRONTAL_PAIRS,
        ),
        Protocol(
            name="smr-still",
            description="SMR up with strict inhibits on slow waves and muscle tension: rewards a quiet, "
            "still body and steady mind, the state used to practise letting impulses pass.",
            reward="smr",
            inhibits=("delta", "theta", "hibeta"),
            sites=("C4", "Cz", "C3"),
        ),
        Protocol(
            name="fm-theta",
            description="Frontal-midline theta up during focused effort: a marker of the brain's "
            "cognitive-control system (anterior cingulate), studied for self-control and working memory.",
            reward="theta",
            inhibits=("delta", "hibeta"),
            sites=("Fz", "FCz", "AFz"),
        ),
        Protocol(
            name="gamma-40",
            description="The brain's 40 Hz response (38-42 Hz) during 40 Hz sound and light stimulation: "
            "a readout of how strongly the rhythm is entraining. Muscle tension also shows up near 40 Hz, "
            "so rewards pause when high-beta rises with it.",
            reward="g40",
            inhibits=("hibeta",),
            sites=("Cz", "Fz", "Oz", "O1", "O2", "Pz"),
        ),
        Protocol(
            name="beta-left",
            description="Beta up over the left prefrontal cortex, which is engaged by goal-directed, "
            "approach-oriented action.",
            reward="beta",
            inhibits=("theta", "hibeta"),
            sites=("F3", "AF3", "FC3", "AF7"),
        ),
    )
}


def get_protocol(name: str) -> Protocol:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"unknown protocol {name!r}; choose from {', '.join(PROTOCOLS)}") from None


def resolve_protocol(ref: str | dict | Protocol) -> Protocol:
    if isinstance(ref, Protocol):
        return ref
    if isinstance(ref, dict):
        return Protocol.from_spec(ref)
    return get_protocol(ref)
