"""The real-time neurofeedback loop.

A session runs a Program: a sequence of phases (baseline, training, rest,
cooldown). Session time is derived from the number of samples received, not
the wall clock, so the same loop drives live hardware and fast offline runs.
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .devices import Device
from .dsp import RingBuffer, StreamingFilter, band_power, peak_to_peak, power_spectrum
from .feedback import FeedbackSink, FeedbackState, SessionAborted
from .programs import Phase, Program, simple_program
from .protocols import BANDS, Protocol
from .stimulation import Stimulus
from .threshold import AdaptiveThreshold, InhibitThreshold


@dataclass
class SessionConfig:
    # baseline_sec / training_sec / target_reward_rate only apply when no program is given.
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
class PhaseSummary:
    name: str
    kind: str
    protocol: str | None
    seconds: float
    metric: float
    reward_fraction: float
    artifact_fraction: float

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in self.__dict__.items()}


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
    program: str = ""
    phases: list[PhaseSummary] = field(default_factory=list)

    @property
    def change(self) -> float:
        if not self.baseline_metric or not np.isfinite(self.baseline_metric):
            return float("nan")
        return (self.training_metric - self.baseline_metric) / abs(self.baseline_metric)

    def format(self) -> str:
        lines = [
            f"Program:         {self.program}" if self.program else None,
            f"Protocol:        {self.protocol}",
            f"Device:          {self.device}",
            f"Channels:        {', '.join(self.channels)}",
            f"Duration:        {self.duration_sec / 60:.1f} min" + ("  (stopped early)" if self.aborted else ""),
            f"Baseline metric: {self.baseline_metric:.4f}",
            f"Training metric: {self.training_metric:.4f}  ({self.change:+.1%} vs baseline)",
            f"Time rewarded:   {self.reward_fraction:.1%}",
            f"Artifacts:       {self.artifact_fraction:.1%} of updates",
        ]
        lines = [line for line in lines if line]
        if len(self.phases) > 2:
            lines.append("Phases:")
            for p in self.phases:
                reward = f"{p.reward_fraction:6.1%} rewarded" if p.kind == "training" else " " * 15
                metric = f"{p.metric:.4f}" if np.isfinite(p.metric) else "   -  "
                lines.append(f"  {p.name:<24} {p.seconds / 60:4.1f} min  {metric}  {reward}")
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

    def to_dict(self) -> dict:
        def clean(v):
            return None if isinstance(v, float) and not np.isfinite(v) else v

        d = {k: clean(v) for k, v in self.__dict__.items() if k != "phases"}
        d["log_path"] = str(self.log_path) if self.log_path else None
        d["phases"] = [p.to_dict() for p in self.phases]
        d["change"] = clean(self.change)
        return d


def _change(on: float, off: float) -> str:
    if not (np.isfinite(on) and np.isfinite(off)) or off == 0:
        return ""
    return f"  ({(on - off) / off:+.1%})"


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@dataclass
class _PhaseStats:
    metrics: list[float] = field(default_factory=list)
    rewarded: int = 0
    updates: int = 0
    artifacts: int = 0
    seconds: float = 0.0


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


class _Trainer:
    """Adaptive reward and inhibit thresholds for one protocol."""

    def __init__(self, protocol: Protocol, cfg: SessionConfig, history: int):
        self.protocol = protocol
        self.threshold = AdaptiveThreshold(cfg.target_reward_rate, protocol.direction, history)
        self.inhibits = {b: InhibitThreshold(cfg.inhibit_rate, history) for b in protocol.inhibits}

    def observe(self, metric: float, inhibit_powers: dict[str, float]) -> None:
        self.threshold.observe(metric)
        for band, th in self.inhibits.items():
            th.observe(inhibit_powers[band])

    def freeze(self) -> None:
        self.threshold.freeze()
        for th in self.inhibits.values():
            th.freeze()


class Session:
    def __init__(
        self,
        device: Device,
        protocol: Protocol | None = None,
        config: SessionConfig | None = None,
        sinks: list[FeedbackSink] | None = None,
        log_path: str | Path | None = None,
        stimulus: Stimulus | None = None,
        program: Program | None = None,
    ):
        self.config = cfg = config or SessionConfig()
        if program is None:
            if protocol is None:
                raise ValueError("a session needs a protocol or a program")
            program = simple_program(protocol, cfg.baseline_sec, cfg.training_sec, cfg.target_reward_rate)
        self.program = program
        self.device = device
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
        problems = program.check_channels(self.channel_names)
        if problems:
            raise ValueError("; ".join(problems) + f". Available channels: {', '.join(self.channel_names)}")

        fs = device.sampling_rate
        n_ch = len(self.channel_idx)
        self._filter = StreamingFilter(fs, n_ch, high=cfg.high_hz, notch=cfg.notch_hz)
        self._buffer = RingBuffer(n_ch, int(cfg.window_sec * fs))
        self._update_samples = max(1, int(cfg.update_sec * fs))
        history = max(4, int(cfg.threshold_history_sec / cfg.update_sec))
        self.trainers = {name: _Trainer(p, cfg, history) for name, p in program.protocols().items()}
        self.protocol = next(iter(self.trainers.values())).protocol  # primary protocol
        self._sustain_updates = max(1, int(round(cfg.sustain_sec / cfg.update_sec)))
        self._inhibit_bands = sorted({b for t in self.trainers.values() for b in t.protocol.inhibits})

        self._phase_idx = 0
        self._phase_start = 0.0
        self._lead_baseline_sec = 0.0
        for phase in program.phases:
            if phase.kind != "baseline":
                break
            self._lead_baseline_sec += phase.seconds
        self._skip = threading.Event()
        self._stop = threading.Event()
        self._frozen = False
        self._streak = 0
        self._samples = 0
        self._since_update = 0
        self._stats = _Stats()
        self._phase_stats = [_PhaseStats() for _ in program.phases]
        self._log_file = None
        self._log = None

    # -- timing -------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return self._samples / self.device.sampling_rate

    @property
    def total_sec(self) -> float:
        return sum(p.seconds for p in self.program.phases)

    @property
    def finished(self) -> bool:
        return self._phase_idx >= len(self.program.phases)

    @property
    def phase(self) -> Phase | None:
        return None if self.finished else self.program.phases[self._phase_idx]

    @property
    def trainer(self) -> _Trainer | None:
        phase = self.phase
        if phase is None or phase.kind != "training":
            return None
        return self.trainers[phase.resolved_protocol().name]

    def _display_trainer(self) -> _Trainer:
        """The protocol to show outside training: the next one coming up, else the last used."""
        phases = self.program.phases
        order = list(range(self._phase_idx, len(phases))) + list(range(min(self._phase_idx, len(phases)) - 1, -1, -1))
        for i in order:
            if phases[i].kind == "training":
                return self.trainers[phases[i].resolved_protocol().name]
        return next(iter(self.trainers.values()))

    def skip_phase(self) -> None:
        """Move on to the next phase at the next update. Safe to call from another thread."""
        self._skip.set()

    def stop(self) -> None:
        """End the session early. Safe to call from another thread."""
        self._stop.set()

    def _sync_phase(self) -> None:
        phases = self.program.phases
        while not self.finished and (
            self._skip.is_set() or self.elapsed - self._phase_start >= phases[self._phase_idx].seconds
        ):
            self._skip.clear()
            self._phase_stats[self._phase_idx].seconds = self.elapsed - self._phase_start
            self._phase_start = self.elapsed
            self._phase_idx += 1
            self._streak = 0
            phase = self.phase
            if phase is not None and phase.kind == "training":
                trainer = self.trainer
                trainer.threshold.set_target(phase.reward_rate)
                if self.config.freeze_thresholds and not self._frozen:
                    self._frozen = True
                    for t in self.trainers.values():
                        t.freeze()

    # -- main loop ----------------------------------------------------------

    def run(self) -> SessionSummary:
        aborted = False
        self._open_log()
        if self.stimulus:
            self.stimulus.open()
        for sink in self.sinks:
            sink.configure(self)
            sink.open()
        first = self.phase
        if first is not None and first.kind == "training":
            self.trainer.threshold.set_target(first.reward_rate)
        try:
            with self.device:
                while not self.finished and not self._stop.is_set():
                    chunk = self.device.read()
                    if chunk.shape[1] == 0:
                        time.sleep(0.005)
                        continue
                    self.process_chunk(chunk)
        except (KeyboardInterrupt, SessionAborted):
            aborted = True
        finally:
            if not self.finished:
                self._phase_stats[self._phase_idx].seconds = self.elapsed - self._phase_start
            aborted = aborted or self._stop.is_set()
            summary = self.summary(aborted)
            for sink in self.sinks:
                sink.finish(summary)
                sink.close()
            if self.stimulus:
                self.stimulus.close()
            self._close_log()
        return summary

    def process_chunk(self, chunk: np.ndarray) -> None:
        filtered = self._filter.process(chunk[self.channel_idx])
        self._buffer.extend(filtered)
        self._samples += chunk.shape[1]
        self._since_update += chunk.shape[1]
        self._sync_phase()
        if self.finished:
            return
        if self._since_update >= self._update_samples and self._buffer.full:
            new = self._since_update
            self._since_update = 0
            self._update(new)

    def _update(self, new_samples: int = 0) -> None:
        cfg = self.config
        phase = self.phase
        kind = phase.kind
        names = self.channel_names
        window = self._buffer.get()
        freqs, psd = power_spectrum(window, self.device.sampling_rate)
        artifact = bool(peak_to_peak(window).max() > cfg.artifact_uv)
        total = band_power(freqs, psd, BANDS["total"]).mean()
        band_powers = {
            b: float(band_power(freqs, psd, rng).mean() / total) if total > 0 else 0.0
            for b, rng in BANDS.items()
            if b != "total"
        }

        # Every protocol is measured each update; thresholds learn during baseline
        # (calibration) and while their own protocol is being trained.
        readings = {
            name: (t.protocol.metric(freqs, psd, names), t.protocol.inhibit_powers(freqs, psd, names))
            for name, t in self.trainers.items()
        }
        active = self.trainer
        shown = active or self._display_trainer()
        metric, inhibit_powers = readings[shown.protocol.name]

        inhibited = [b for b, th in shown.inhibits.items() if th.triggered(inhibit_powers[b])]
        passing = shown.threshold.passes(metric) and not inhibited and not artifact
        self._streak = self._streak + 1 if passing and active else 0
        reward = active is not None and self._streak >= self._sustain_updates

        stim_on = self.stimulus.step(self.elapsed, self._lead_baseline_sec) if self.stimulus else False
        stim_on = stim_on or phase.binaural is not None
        stim_power = self._stim_power(freqs, psd, total, phase)

        st = self._stats
        ps = self._phase_stats[self._phase_idx]
        st.updates += 1
        ps.updates += 1
        if artifact:
            st.artifacts += 1
            ps.artifacts += 1
        else:
            # Keep artifacts out of the threshold history so a blink doesn't move the goalposts.
            if kind == "baseline":
                for name, t in self.trainers.items():
                    t.observe(*readings[name])
            elif active is not None:
                active.observe(metric, inhibit_powers)
            if np.isfinite(metric):
                ps.metrics.append(metric)
            primary = readings[self.protocol.name][0]
            if kind == "baseline" and np.isfinite(primary):
                st.baseline.append(primary)
            elif active is not None and active.protocol is self.protocol and np.isfinite(metric):
                st.training.append(metric)
            if self.stimulus and kind == "training":
                (st.metric_on if stim_on else st.metric_off).append(metric)
                (st.stim_power_on if stim_on else st.stim_power_off).append(stim_power)

        if kind == "training":
            st.training_updates += 1
            st.rewarded += reward
            ps.rewarded += reward

        state = FeedbackState(
            t=self.elapsed,
            phase=kind,
            metric=metric,
            threshold=shown.threshold.value,
            reward=reward,
            intensity=0.0 if (artifact or inhibited) else shown.threshold.intensity(metric),
            inhibited=inhibited,
            artifact=artifact,
            reward_fraction=st.rewarded / st.training_updates if st.training_updates else 0.0,
            band_powers=band_powers,
            stim_on=stim_on,
            samples=window[:, -min(new_samples, window.shape[1]) :] if new_samples else None,
            phase_name=phase.name,
            phase_index=self._phase_idx,
            phase_count=len(self.program.phases),
            phase_elapsed=self.elapsed - self._phase_start,
            phase_duration=phase.seconds,
            session_duration=self.total_sec,
            guidance=phase.guidance,
            binaural=phase.binaural,
            protocol=shown.protocol.name,
            metric_label=shown.protocol.metric_label,
            direction=shown.protocol.direction,
        )
        for sink in self.sinks:
            sink.update(state)
        self._write_log(state, inhibit_powers, stim_power)

    def _stim_power(self, freqs: np.ndarray, psd: np.ndarray, total: float, phase: Phase) -> float:
        """Power at the stimulation frequency (±1 Hz) relative to 1-40 Hz power."""
        f = self.stimulus.spec.freq if self.stimulus else phase.binaural
        if f is None or total <= 0:
            return float("nan")
        return float(band_power(freqs, psd, (f - 1.0, f + 1.0)).mean() / total)

    # -- results ------------------------------------------------------------

    def summary(self, aborted: bool = False) -> SessionSummary:
        st = self._stats
        phases = [
            PhaseSummary(
                name=p.name,
                kind=p.kind,
                protocol=p.resolved_protocol().name if p.protocol is not None else None,
                seconds=ps.seconds,
                metric=_mean(ps.metrics),
                reward_fraction=ps.rewarded / ps.updates if ps.updates else 0.0,
                artifact_fraction=ps.artifacts / ps.updates if ps.updates else 0.0,
            )
            for p, ps in zip(self.program.phases, self._phase_stats)
            if ps.updates or ps.seconds
        ]
        return SessionSummary(
            protocol=self.protocol.name,
            device=self.device.description,
            channels=self.channel_names,
            baseline_metric=_mean(st.baseline),
            training_metric=_mean(st.training),
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
            program=self.program.name,
            phases=phases,
        )

    def _open_log(self) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", newline="")
        self._log = csv.writer(self._log_file)
        self._log.writerow(
            ["t", "phase", "phase_name", "protocol", "metric", "threshold", "reward", "intensity", "artifact", "inhibited"]
            + [f"{b}_power" for b in self._inhibit_bands]
            + ["stim_on", "stim_power"]
        )

    def _write_log(self, s: FeedbackState, inhibit_powers: dict[str, float], stim_power: float) -> None:
        if not self._log:
            return
        self._log.writerow(
            [
                f"{s.t:.3f}",
                s.phase,
                s.phase_name,
                s.protocol,
                f"{s.metric:.6f}",
                f"{s.threshold:.6f}",
                int(s.reward),
                f"{s.intensity:.3f}",
                int(s.artifact),
                "|".join(s.inhibited),
            ]
            + [f"{inhibit_powers[b]:.4f}" if b in inhibit_powers else "" for b in self._inhibit_bands]
            + [int(s.stim_on), f"{stim_power:.6f}"]
        )

    def _close_log(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log = None
