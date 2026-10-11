import csv
import io

import numpy as np
import pytest

from neuro.cli import main
from neuro.devices import BrainFlowDevice, SimulatedDevice
from neuro.feedback import ConsoleFeedback, FeedbackSink
from neuro.protocols import get_protocol
from neuro.session import Session, SessionConfig


class Recorder(FeedbackSink):
    def __init__(self):
        self.states = []

    def update(self, state):
        self.states.append(state)


def fast_device(**kw):
    return SimulatedDevice(realtime=False, chunk_size=25, seed=42, **kw)


def test_simulated_device_shape():
    dev = fast_device(n_channels=3)
    with dev:
        chunk = dev.read()
    assert chunk.shape == (3, 25)


def test_session_runs_and_logs(tmp_path):
    rec = Recorder()
    log = tmp_path / "s.csv"
    cfg = SessionConfig(baseline_sec=20, training_sec=60, target_reward_rate=0.6)
    summary = Session(fast_device(), get_protocol("alpha"), cfg, [rec], log).run()

    assert summary.duration_sec == pytest.approx(80, abs=0.2)
    assert not summary.aborted
    phases = {s.phase for s in rec.states}
    assert phases == {"baseline", "training"}
    assert not any(s.reward for s in rec.states if s.phase == "baseline")
    # Sustain requirement means reward rate ends up below the raw threshold-pass target.
    assert 0.3 < summary.reward_fraction < 0.7
    assert all(0.0 <= s.intensity <= 1.0 for s in rec.states)

    rows = list(csv.DictReader(log.open()))
    assert len(rows) == len(rec.states)
    assert {"metric", "threshold", "reward", "delta_power", "hibeta_power"} <= set(rows[0])


def test_rewards_track_simulated_alpha():
    """Simulator alpha waxes and wanes; rewards should cluster where alpha is high."""
    rec = Recorder()
    cfg = SessionConfig(baseline_sec=20, training_sec=80, sustain_sec=0.25)
    Session(fast_device(), get_protocol("alpha"), cfg, [rec]).run()
    training = [s for s in rec.states if s.phase == "training" and not s.artifact]
    alpha_drive = np.array([np.sin(2 * np.pi * s.t / 20) for s in training])
    rewards = np.array([s.reward for s in training], dtype=float)
    assert np.corrcoef(alpha_drive, rewards)[0, 1] > 0.3


def test_artifacts_withhold_reward():
    rec = Recorder()
    cfg = SessionConfig(baseline_sec=5, training_sec=10, artifact_uv=1.0)
    summary = Session(fast_device(), get_protocol("alpha"), cfg, [rec]).run()
    assert summary.artifact_fraction == 1.0
    assert summary.reward_fraction == 0.0


def test_channel_selection():
    cfg = SessionConfig(baseline_sec=3, training_sec=3, channels=["SIM2", "SIM4"])
    s = Session(fast_device(), get_protocol("smr"), cfg)
    assert s.channel_names == ["SIM2", "SIM4"]
    with pytest.raises(ValueError):
        Session(fast_device(), get_protocol("smr"), SessionConfig(channels=["Cz"]))


def test_console_feedback_renders():
    out = io.StringIO()
    cfg = SessionConfig(baseline_sec=3, training_sec=3)
    Session(fast_device(), get_protocol("theta-beta"), cfg, [ConsoleFeedback(out)]).run()
    text = out.getvalue()
    assert "calibrating" in text and "training" in text


def test_cli_lists(capsys):
    assert main(["protocols"]) == 0
    assert "theta-beta" in capsys.readouterr().out
    assert main(["devices"]) == 0
    assert "sim" in capsys.readouterr().out


def test_cli_unknown_board(capsys):
    pytest.importorskip("brainflow")
    assert main(["run", "--device", "not_a_board", "--no-log"]) == 1
    assert "unknown BrainFlow board" in capsys.readouterr().err


def test_brainflow_synthetic_board():
    pytest.importorskip("brainflow")
    dev = BrainFlowDevice("synthetic")
    assert dev.sampling_rate > 0
    assert len(dev.channel_names) > 0
    # Synthetic-board amplitude grows with channel index; the first channels look like real EEG.
    cfg = SessionConfig(baseline_sec=2.5, training_sec=1.0, window_sec=1.0, channels=dev.channel_names[:2])
    summary = Session(dev, get_protocol("alpha"), cfg).run()
    assert summary.updates > 0
    assert np.isfinite(summary.training_metric)
