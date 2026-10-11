"""Streaming signal processing: filtering, buffering and spectral band power."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

try:
    _trapezoid = np.trapezoid
except AttributeError:  # numpy < 2.0
    _trapezoid = np.trapz


class StreamingFilter:
    """Band-pass plus optional notch, applied chunk by chunk with carried state.

    Processing a recording in arbitrary chunks gives the same output as
    processing it in one go, which is what makes it safe for live data.
    """

    def __init__(
        self,
        sampling_rate: float,
        n_channels: int,
        low: float = 1.0,
        high: float = 40.0,
        notch: float | None = 60.0,
        order: int = 4,
    ):
        nyq = sampling_rate / 2
        high = min(high, nyq * 0.95)
        sections = [butter(order, [low, high], btype="bandpass", fs=sampling_rate, output="sos")]
        if notch and notch < nyq:
            b, a = iirnotch(notch, Q=30.0, fs=sampling_rate)
            sections.append(tf2sos(b, a))
        self.sos = np.vstack(sections)
        self.n_channels = n_channels
        self._zi: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.shape[1] == 0:
            return chunk
        if self._zi is None:
            # Start in steady state for the first sample to avoid a large onset transient.
            self._zi = sosfilt_zi(self.sos)[:, None, :] * chunk[:, 0][None, :, None]
        out, self._zi = sosfilt(self.sos, chunk, axis=1, zi=self._zi)
        return out


class RingBuffer:
    """Fixed-length multi-channel buffer holding the most recent samples."""

    def __init__(self, n_channels: int, length: int):
        self._data = np.zeros((n_channels, length))
        self._filled = 0

    @property
    def full(self) -> bool:
        return self._filled >= self._data.shape[1]

    def extend(self, chunk: np.ndarray) -> None:
        n = chunk.shape[1]
        length = self._data.shape[1]
        if n >= length:
            self._data[:] = chunk[:, -length:]
        else:
            self._data = np.roll(self._data, -n, axis=1)
            self._data[:, -n:] = chunk
        self._filled = min(length, self._filled + n)

    def get(self) -> np.ndarray:
        return self._data[:, -self._filled :] if self._filled else self._data[:, :0]


def power_spectrum(window: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (µV²/Hz) per channel, ~1 Hz resolution."""
    nperseg = min(window.shape[-1], int(sampling_rate))
    return welch(window, fs=sampling_rate, nperseg=nperseg, axis=-1)


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Absolute power (µV²) in [low, high) Hz for each channel."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        # Band narrower than the frequency resolution: take nearest bins.
        idx = np.argsort(np.abs(freqs - (lo + hi) / 2))[:2]
        mask = np.zeros_like(freqs, dtype=bool)
        mask[idx] = True
    return _trapezoid(psd[..., mask], freqs[mask], axis=-1)


def peak_to_peak(window: np.ndarray) -> np.ndarray:
    return np.ptp(window, axis=-1)
