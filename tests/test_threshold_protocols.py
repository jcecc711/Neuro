import numpy as np
import pytest

from neuro.dsp import power_spectrum
from neuro.protocols import PROTOCOLS, get_protocol
from neuro.threshold import AdaptiveThreshold, InhibitThreshold

FS = 250.0


@pytest.mark.parametrize("direction", [1, -1])
def test_adaptive_threshold_hits_target_rate(direction):
    rng = np.random.default_rng(1)
    th = AdaptiveThreshold(target_rate=0.7, direction=direction, history=500)
    values = rng.normal(size=500)
    for v in values:
        th.observe(v)
    rate = np.mean([th.passes(v) for v in values])
    assert rate == pytest.approx(0.7, abs=0.02)


def test_threshold_not_ready_never_passes():
    th = AdaptiveThreshold(0.5)
    assert not th.passes(1e9)
    assert th.intensity(1.0) == 0.0


def test_frozen_threshold_stops_adapting():
    th = AdaptiveThreshold(0.5, history=10)
    for v in range(10):
        th.observe(v)
    before = th.value
    th.freeze()
    th.observe(1000)
    assert th.value == before


def test_inhibit_triggers_on_spikes():
    th = InhibitThreshold(trigger_rate=0.1, history=100)
    for v in np.linspace(0, 1, 100):
        th.observe(v)
    assert th.triggered(5.0)
    assert not th.triggered(0.2)


def _psd_of(freqs_amps):
    t = np.arange(int(4 * FS)) / FS
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in freqs_amps)
    rng = np.random.default_rng(0)
    x = x + 0.5 * rng.standard_normal(t.size)
    return power_spectrum(x[None], FS)


def test_alpha_metric_increases_with_alpha():
    p = get_protocol("alpha")
    low = p.metric(*_psd_of([(10, 2), (20, 5)]))
    high = p.metric(*_psd_of([(10, 10), (20, 5)]))
    assert 0 < low < high < 1


def test_theta_beta_ratio_trains_down():
    p = get_protocol("theta-beta")
    assert p.direction == -1
    ratio = p.metric(*_psd_of([(6, 10), (18, 5)]))
    assert ratio == pytest.approx(4.0, rel=0.2)


def test_all_protocols_produce_finite_metrics():
    freqs, psd = _psd_of([(6, 5), (10, 5), (13, 5), (18, 5), (25, 5)])
    for p in PROTOCOLS.values():
        assert np.isfinite(p.metric(freqs, psd))
        assert set(p.inhibit_powers(freqs, psd)) == set(p.inhibits)


def test_unknown_protocol():
    with pytest.raises(ValueError):
        get_protocol("nope")
