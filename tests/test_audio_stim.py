import numpy as np
import pytest

from nova.audio import AudioEngine, BinauralVoice, NoiseVoice, PulsedToneVoice, ToneVoice
from nova.devices import SimulatedDevice
from nova.feedback import FeedbackState, SonifyFeedback, make_sinks
from nova.protocols import get_protocol
from nova.session import Session, SessionConfig
from nova.stimulation import Stimulus, StimulusSpec

SR = 8000


class SilentEngine(AudioEngine):
    """Mixes voices like the real engine but never opens an audio device."""

    def __init__(self):
        super().__init__(sample_rate=SR)
        self.started = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.started -= 1


def peak_freq(x, sr=SR):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[spectrum.argmax()]


def test_binaural_beat_differs_between_ears():
    v = BinauralVoice(carrier=200, beat=10, level=1.0)
    out = v.render(SR * 2, SR)
    assert peak_freq(out[:, 0]) == pytest.approx(200, abs=1)
    assert peak_freq(out[:, 1]) == pytest.approx(210, abs=1)


def test_pulsed_tone_envelope_at_rate():
    v = PulsedToneVoice(carrier=500, rate=40, level=1.0)
    out = v.render(SR * 2, SR)[:, 0]
    envelope = np.abs(out) ** 2  # demodulate
    envelope -= envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 1 / SR)
    low = freqs < 200
    assert freqs[low][spectrum[low].argmax()] == pytest.approx(40, abs=1)


def test_voices_are_continuous_across_blocks():
    """Rendering in blocks must not introduce phase jumps (audible clicks)."""
    a = ToneVoice(440, level=0.5)
    b = ToneVoice(440, level=0.5)
    whole = a.render(1000, SR)
    pieces = np.concatenate([b.render(n, SR) for n in (100, 333, 567)])
    np.testing.assert_allclose(whole, pieces, atol=1e-9)


def test_level_changes_are_ramped():
    v = ToneVoice(440, level=0.0)
    v.target = 1.0
    out = v.render(SR, SR)[:, 0]
    assert np.abs(out[:100]).max() < 0.05  # starts near silent, no step


def test_engine_mixes_and_clips():
    eng = AudioEngine(sample_rate=SR)
    for _ in range(5):
        eng.add(ToneVoice(220, level=1.0))
    eng.add(NoiseVoice(level=1.0, seed=0))
    out = eng.render(512)
    assert out.shape == (512, 2)
    assert np.abs(out).max() <= 1.0


def test_stimulus_spec_defaults_and_validation():
    assert StimulusSpec.create("binaural").freq == 10
    assert StimulusSpec.create("pulsed").freq == 40
    with pytest.raises(ValueError):
        StimulusSpec.create("laser")
    with pytest.raises(ValueError):
        StimulusSpec.create("pulsed", schedule="sometimes")


def test_stimulus_schedules():
    blocks = StimulusSpec.create("pulsed", schedule="blocks", block_sec=10)
    assert not blocks.active_at(5, baseline_sec=10)
    assert blocks.active_at(12, 10)
    assert not blocks.active_at(22, 10)
    assert blocks.active_at(31, 10)
    assert StimulusSpec.create("pulsed", schedule="always").active_at(0, 10)
    training = StimulusSpec.create("pulsed", schedule="training")
    assert not training.active_at(9, 10) and training.active_at(100, 10)


def test_sonify_levels_follow_relative_change():
    s = SonifyFeedback(SilentEngine())
    steady = {"delta": 0.3, "theta": 0.2, "alpha": 0.2, "smr": 0.1, "beta": 0.1, "hibeta": 0.05}
    for _ in range(200):
        s.update(FeedbackState(0, "training", 0, 0, False, 0, band_powers=steady))
    typical = s.voices["alpha"].target
    s.update(FeedbackState(0, "training", 0, 0, False, 0, band_powers={**steady, "alpha": 0.5}))
    assert s.voices["alpha"].target > 2 * typical
    assert s.voices["theta"].target == pytest.approx(typical, rel=0.05)
    s.update(FeedbackState(0, "training", 0, 0, False, 0, artifact=True, band_powers=steady))
    assert all(v.target == 0 for v in s.voices.values())


def test_make_sinks_shares_engine():
    eng = SilentEngine()
    sinks = make_sinks(["audio", "sonify"], eng)
    for sink in sinks:
        sink.open()
    assert eng.started == 2
    for sink in sinks:
        sink.close()
    assert eng.started == 0


class RespondingDevice(SimulatedDevice):
    """Simulator whose EEG picks up a 40 Hz response while the stimulus plays."""

    def __init__(self, stimulus):
        super().__init__(realtime=False, chunk_size=25, seed=3)
        self.stimulus = stimulus

    def _generate(self, count):
        start = self._n
        x = super()._generate(count)
        if self.stimulus.on:
            t = (start + np.arange(count)) / self.sampling_rate
            x += 3.0 * np.sin(2 * np.pi * 40.0 * t)
        return x


def test_session_measures_stimulus_response(tmp_path):
    spec = StimulusSpec.create("pulsed", schedule="blocks", block_sec=10)
    stim = Stimulus(spec, SilentEngine())
    cfg = SessionConfig(baseline_sec=10, training_sec=40, high_hz=45.0, notch_hz=60.0)
    log = tmp_path / "stim.csv"
    summary = Session(RespondingDevice(stim), get_protocol("alpha"), cfg, log_path=log, stimulus=stim).run()

    assert summary.stimulus and "40 Hz" in summary.stimulus
    assert summary.stim_power_on > 3 * summary.stim_power_off
    assert np.isfinite(summary.metric_stim_on) and np.isfinite(summary.metric_stim_off)
    assert "relative power on / off" in summary.format()
    header = log.read_text().splitlines()[0]
    assert header.endswith("stim_on,stim_power")


def test_cli_stim_writes_wav(tmp_path):
    from scipy.io import wavfile

    from nova.cli import main

    out = tmp_path / "beat.wav"
    assert main(["stim", "--stim", "binaural", "--seconds", "5", "--out", str(out)]) == 0
    sr, x = wavfile.read(out)
    assert x.shape == (5 * sr, 2)
    assert abs(x[:100]).max() < 50  # faded in


def test_cli_run_with_stimulus(capsys):
    from nova.cli import main

    args = ["run", "--baseline", "3", "--duration", "4", "--stim", "pulsed", "--stim-block", "2", "--no-log"]
    # Real audio output may be unavailable (CI); the run must then fail cleanly, not crash.
    code = main(args)
    out = capsys.readouterr()
    if code == 0:
        assert "40 Hz relative power on / off" in out.out
    else:
        assert "audio" in out.err
