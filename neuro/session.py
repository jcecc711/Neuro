"""The real-time neurofeedback loop.

Session time is derived from the number of samples received, not the wall
clock, so the same loop drives live hardware and fast offline simulation.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .devices import Device
from .dsp import RingBuffer, StreamingFilter, band_power, peak_to_peak, power_spectrum
from .feedback import FeedbackSink, FeedbackState, SessionAborted
from .protocols import BANDS, Protocol
from .stimulation import Stimulus
from .threshold import AdaptiveThreshold, InhibitThreshold


@dataclass
class SessionConfig:
    baseline_sec: float = 60.0
    training_sec: float = 300.0
    window_sec: float = 2.0
    update_sec: float = 0.25
    target_reward_rate: float = 0.6
    inhibit_rate: float = 0.1
    threshold_history_sec: float = 30.0
    freeze_thresholds: bool = False
    sustain_sec: float = 0.5
    artifact_uv: float = 200.0
    notch_hz: float | None = 60.0
    high_hz: float = 40.0  # band-pass upper edge; raise it to see 40 Hz stimulation responses
    channels: list[str] | None = None  # None = all channels


@dataclass
class SessionSummary:
    protocol: str
    device: str
    channels: list[str]
    baseline_metric: float
    training_metric: float
    reward_fraction: float
    artifact_fraction: float
    updates: int
    duration_sec: float
    log_path: Path | None = None
    aborted: bool = False
    stimulus: str | None = None
    stim_freq: float | None = None
    metric_stim_on: float = float("nan")
    metric_stim_off: float = float("nan")
    stim_power_on: float = float("nan")
    stim_power_off: float = float("nan")

    def format(self) -> str:
        change = (
            (self.training_metric - self.baseline_metric) / self.baseline_metric
            if self.baseline_metric
            else float("nan")
        )
        lines = [
            f"Protocol:        {self.protocol}",
            f"Device:          {self.device}",
            f"Channels:        {', '.join(self.channels)}",
            f"Duration:        {self.duration_sec:.1f} s" + ("  (stopped early)" if self.aborted else ""),
            f"Baseline metric: {self.baseline_metric:.4f}",
            f"Training metric: {self.training_metric:.4f}  ({change:+.1%} vs baseline)",
            f"Time rewarded:   {self.reward_fraction:.1%}",
            f"Artifacts:       {self.artifact_fraction:.1%} of updates",
        ]
        if self.stimulus:
            lines.append(f"Stimulus:        {self.stimulus}")
            lines.append(
                f"  metric with sound on / off:          {self.metric_stim_on:.4f} / {self.metric_stim_off:.4f}"
            )
            lines.append(
                f"  {self.stim_freq:g} Hz relative power on / off:   "
                f"{self.stim_power_on:.4f} / {self.stim_power_off:.4f}"
                + _change(self.stim_power_on, self.stim_power_off)
            )
        if self.log_path:
            lines.append(f"Log:             {self.log_path}")
        return "\n".join(lines)


def _change(on: float, off: float) -> str:
    if not (np.isfinite(on) and np.isfinite(off)) or off == 0:
        return ""
    return f"  ({(on - off) / off:+.1%})"


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@dataclass
class _Stats:
    baseline: list[float] = field(default_factory=list)
    training: list[float] = field(default_factory=list)
    rewarded: int = 0
    training_updates: int = 0
    artifacts: int = 0
    updates: int = 0
    metric_on: list[float] = field(default_factory=list)
    metric_off: list[float] = field(default_factory=list)
    stim_power_on: list[float] = field(default_factory=list)
    stim_power_off: list[float] = field(default_factory=list)


class Session:
    def __init__(
        self,
        device: Device,
        protocol: Protocol,
        config: SessionConfig | None = None,
        sinks: list[FeedbackSink] | None = None,
        log_path: str | Path | None = None,
        stimulus: Stimulus | None = None,
    ):
        self.device = device
        self.protocol = protocol
        self.config = cfg = config or SessionConfig()
        self.sinks = sinks or []
        self.log_path = Path(log_path) if log_path else None
        self.stimulus = stimulus

        names = device.channel_names
        if cfg.channels:
            missing = [c for c in cfg.channels if c not in names]
            if missing:
                raise ValueError(f"channels {missing} not on device; available: {', '.join(names)}")
            self.channel_idx = [names.index(c) for c in cfg.channels]
        else:
            self.channel_idx = list(range(len(names)))
        self.channel_names = [names[i] for i in self.channel_idx]

        fs = device.sampling_rate
        n_ch = len(self.channel_idx)
        self._filter = StreamingFilter(fs, n_ch, high=cfg.high_hz, notch=cfg.notch_hz)
        self._buffer = RingBuffer(n_ch, int(cfg.window_sec * fs))
        self._update_samples = max(1, int(cfg.update_sec * fs))
        history = max(4, int(cfg.threshold_history_sec / cfg.update_sec))
        self.threshold = AdaptiveThreshold(cfg.target_reward_rate, protocol.direction, history)
        self.inhibits = {b: InhibitThreshold(cfg.inhibit_rate, history) for b in protocol.inhibits}
        self._sustain_updates = max(1, int(round(cfg.sustain_sec / cfg.update_sec)))
        self._streak = 0
        self._samples = 0
        self._since_update = 0
        self._stats = _Stats()
        self._log_file = None
        self._log = None

    @property
    def elapsed(self) -> float:
        return self._samples / self.device.sampling_rate

    @property
    def total_sec(self) -> float:
        return self.config.baseline_sec + self.config.training_sec

    def run(self) -> SessionSummary:
        aborted = False
        self._open_log()
        if self.stimulus:
            self.stimulus.open()
        for sink in self.sinks:
            sink.open()
        try:
            with self.device:
                while self.elapsed < self.total_sec:
                    chunk = self.device.read()
                    if chunk.shape[1] == 0:
                        time.sleep(0.005)
                        continue
                    self.process_chunk(chunk)
        except (KeyboardInterrupt, SessionAborted):
            aborted = True
        finally:
            for sink in self.sinks:
                sink.close()
            if self.stimulus:
                self.stimulus.close()
            self._close_log()
        return self.summary(aborted)

    def process_chunk(self, chunk: np.ndarray) -> None:
        filtered = self._filter.process(chunk[self.channel_idx])
        self._buffer.extend(filtered)
        self._samples += chunk.shape[1]
        self._since_update += chunk.shape[1]
        if self._since_update >= self._update_samples and self._buffer.full:
            self._since_update = 0
            self._update()

    def _update(self) -> None:
        cfg = self.config
        window = self._buffer.get()
        freqs, psd = power_spectrum(window, self.device.sampling_rate)
        metric = self.protocol.metric(freqs, psd)
        inhibit_powers = self.protocol.inhibit_powers(freqs, psd)
        artifact = bool(peak_to_peak(window).max() > cfg.artifact_uv)
        phase = "baseline" if self.elapsed < cfg.baseline_sec else "training"
        total = band_power(freqs, psd, BANDS["total"]).mean()
        band_powers = {
            b: float(band_power(freqs, psd, rng).mean() / total) if total > 0 else 0.0
            for b, rng in BANDS.items()
            if b != "total"
        }
        stim_on = self.stimulus.step(self.elapsed, cfg.baseline_sec) if self.stimulus else False
        stim_power = self._stim_power(freqs, psd, total)
        st = self._stats
        st.updates += 1

        if phase == "training" and cfg.freeze_thresholds and not self.threshold.frozen:
            self.threshold.freeze()
            for th in self.inhibits.values():
                th.freeze()

        inhibited = [b for b, th in self.inhibits.items() if th.triggered(inhibit_powers[b])]
        passing = self.threshold.passes(metric) and not inhibited and not artifact
        self._streak = self._streak + 1 if passing else 0
        reward = phase == "training" and self._streak >= self._sustain_updates

        if artifact:
            st.artifacts += 1
        else:
            # Keep artifacts out of the threshold history so a blink doesn't move the goalposts.
            self.threshold.observe(metric)
            for b, th in self.inhibits.items():
                th.observe(inhibit_powers[b])
            (st.baseline if phase == "baseline" else st.training).append(metric)
            if self.stimulus and phase == "training":
                (st.metric_on if stim_on else st.metric_off).append(metric)
                (st.stim_power_on if stim_on else st.stim_power_off).append(stim_power)

        if phase == "training":
            st.training_updates += 1
            st.rewarded += reward

        state = FeedbackState(
            t=self.elapsed,
            phase=phase,
            metric=metric,
            threshold=self.threshold.value,
            reward=reward,
            intensity=0.0 if (artifact or inhibited) else self.threshold.intensity(metric),
            inhibited=inhibited,
            artifact=artifact,
            reward_fraction=st.rewarded / st.training_updates if st.training_updates else 0.0,
            band_powers=band_powers,
            stim_on=stim_on,
        )
        for sink in self.sinks:
            sink.update(state)
        self._write_log(state, inhibit_powers, stim_power)

    def _stim_power(self, freqs: np.ndarray, psd: np.ndarray, total: float) -> float:
        """Power at the stimulation frequency (±1 Hz) relative to 1-40 Hz power."""
        if not self.stimulus or total <= 0:
            return float("nan")
        f = self.stimulus.spec.freq
        return float(band_power(freqs, psd, (f - 1.0, f + 1.0)).mean() / total)

    def summary(self, aborted: bool = False) -> SessionSummary:
        st = self._stats
        return SessionSummary(
            protocol=self.protocol.name,
            device=self.device.description,
            channels=self.channel_names,
            baseline_metric=float(np.mean(st.baseline)) if st.baseline else float("nan"),
            training_metric=float(np.mean(st.training)) if st.training else float("nan"),
            reward_fraction=st.rewarded / st.training_updates if st.training_updates else 0.0,
            artifact_fraction=st.artifacts / st.updates if st.updates else 0.0,
            updates=st.updates,
            duration_sec=self.elapsed,
            log_path=self.log_path,
            aborted=aborted,
            stimulus=self.stimulus.spec.description if self.stimulus else None,
            stim_freq=self.stimulus.spec.freq if self.stimulus else None,
            metric_stim_on=_mean(st.metric_on),
            metric_stim_off=_mean(st.metric_off),
            stim_power_on=_mean(st.stim_power_on),
            stim_power_off=_mean(st.stim_power_off),
        )

    def _open_log(self) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", newline="")
        self._log = csv.writer(self._log_file)
        self._log.writerow(
            ["t", "phase", "metric", "threshold", "reward", "intensity", "artifact", "inhibited"]
            + [f"{b}_power" for b in self.protocol.inhibits]
            + ["stim_on", "stim_power"]
        )

    def _write_log(self, s: FeedbackState, inhibit_powers: dict[str, float], stim_power: float) -> None:
        if not self._log:
            return
        self._log.writerow(
            [
                f"{s.t:.3f}",
                s.phase,
                f"{s.metric:.6f}",
                f"{s.threshold:.6f}",
                int(s.reward),
                f"{s.intensity:.3f}",
                int(s.artifact),
                "|".join(s.inhibited),
            ]
            + [f"{inhibit_powers[b]:.4f}" for b in self.protocol.inhibits]
            + [int(s.stim_on), f"{stim_power:.6f}"]
        )

    def _close_log(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log = None
