import numpy as np

from neuro.dsp import RingBuffer, StreamingFilter, band_power, power_spectrum

FS = 250.0


def sine(freq, seconds=4.0, amp=10.0, n_ch=2):
    t = np.arange(int(seconds * FS)) / FS
    return np.tile(amp * np.sin(2 * np.pi * freq * t), (n_ch, 1))


def test_band_power_concentrates_in_correct_band():
    freqs, psd = power_spectrum(sine(10.0), FS)
    alpha = band_power(freqs, psd, (8, 12))
    beta = band_power(freqs, psd, (15, 22))
    assert np.all(alpha > 50 * beta)
    # Power of a sine is amp²/2.
    np.testing.assert_allclose(alpha, 50.0, rtol=0.1)


def test_notch_removes_line_noise_but_keeps_alpha():
    x = sine(10.0, amp=10.0) + sine(60.0, amp=20.0)
    f = StreamingFilter(FS, 2, notch=60.0)
    y = f.process(x)[:, int(FS):]  # drop settling
    freqs, psd = power_spectrum(y, FS)
    assert band_power(freqs, psd, (58, 62)).max() < 1.0
    np.testing.assert_allclose(band_power(freqs, psd, (8, 12)), 50.0, rtol=0.15)


def test_chunked_filtering_matches_one_shot():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 1000))
    whole = StreamingFilter(FS, 3).process(x)
    f = StreamingFilter(FS, 3)
    pieces = [f.process(x[:, i : i + 37]) for i in range(0, 1000, 37)]
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), whole, atol=1e-10)


def test_ring_buffer_keeps_most_recent():
    rb = RingBuffer(1, 5)
    assert not rb.full
    rb.extend(np.arange(3.0)[None])
    np.testing.assert_array_equal(rb.get(), [[0, 1, 2]])
    rb.extend(np.arange(3.0, 7.0)[None])
    assert rb.full
    np.testing.assert_array_equal(rb.get(), [[2, 3, 4, 5, 6]])
    rb.extend(np.arange(10.0, 20.0)[None])
    np.testing.assert_array_equal(rb.get(), [[15, 16, 17, 18, 19]])
