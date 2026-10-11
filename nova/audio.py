"""Audio synthesis: a single stereo output stream that mixes any number of voices.

Feedback tones, sonification and stimulation all render through one engine so
they share an output device (built-in speakers, wired or Bluetooth headphones
such as AirPods) without fighting over it. Voices are plain numpy generators,
so they can be rendered and tested without any audio hardware.
"""

from __future__ import annotations

import math
import threading

import numpy as np
from scipy.signal import lfilter

TWO_PI = 2 * math.pi


class Voice:
    """A sound source with a smoothly ramped output level (0..1)."""

    def __init__(self, level: float = 0.0):
        self.target = level
        self._amp = level

    def _envelope(self, frames: int) -> np.ndarray:
        # Ramp across the block so level changes never click.
        env = np.linspace(self._amp, self.target, frames, dtype=np.float64)
        self._amp = self.target
        return env

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        """Return a (frames, 2) stereo block."""
        raise NotImplementedError


class _Oscillator:
    def __init__(self, freq: float):
        self.freq = freq
        self._phase = 0.0

    def __call__(self, frames: int, sample_rate: int) -> np.ndarray:
        step = TWO_PI * self.freq / sample_rate
        phases = self._phase + step * np.arange(frames)
        self._phase = float((self._phase + step * frames) % TWO_PI)
        return np.sin(phases)


class ToneVoice(Voice):
    """Sine tone with optional soft harmonics and stereo pan (-1 left .. +1 right)."""

    def __init__(self, freq: float, pan: float = 0.0, harmonics: tuple[float, ...] = (), level: float = 0.0):
        super().__init__(level)
        self._osc = [_Oscillator(freq)] + [_Oscillator(freq * (i + 2)) for i in range(len(harmonics))]
        self._weights = np.array((1.0, *harmonics))
        self._weights /= self._weights.sum()
        angle = (pan + 1) * math.pi / 4  # constant-power pan
        self._gains = np.array([math.cos(angle), math.sin(angle)])

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        mono = sum(w * osc(frames, sample_rate) for w, osc in zip(self._weights, self._osc))
        return (self._envelope(frames) * mono)[:, None] * self._gains


class NoiseVoice(Voice):
    """Soft band-limited hiss, used to make muscle tension audible."""

    def __init__(self, brightness: float = 0.3, level: float = 0.0, seed: int | None = None):
        super().__init__(level)
        self._rng = np.random.default_rng(seed)
        # One-pole low-pass: y[n] = y[n-1] + brightness * (x[n] - y[n-1]).
        self._b = [brightness]
        self._a = [1.0, brightness - 1.0]
        self._zi = np.zeros(1)

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        white = self._rng.uniform(-1, 1, frames)
        out, self._zi = lfilter(self._b, self._a, white, zi=self._zi)
        return (self._envelope(frames) * out)[:, None] * np.array([0.7, 0.7])


class BinauralVoice(Voice):
    """Carrier in the left ear, carrier + beat in the right. Headphones required."""

    def __init__(self, carrier: float, beat: float, level: float = 0.0):
        super().__init__(level)
        self.carrier = carrier
        self.beat = beat
        self._left = _Oscillator(carrier)
        self._right = _Oscillator(carrier + beat)

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        env = self._envelope(frames)
        return np.column_stack([env * self._left(frames, sample_rate), env * self._right(frames, sample_rate)])


class PulsedToneVoice(Voice):
    """Carrier tone amplitude-modulated at `rate` Hz (e.g. 40 Hz auditory stimulation)."""

    def __init__(self, carrier: float, rate: float, depth: float = 1.0, level: float = 0.0):
        super().__init__(level)
        self.carrier = carrier
        self.rate = rate
        self.depth = depth
        self._tone = _Oscillator(carrier)
        self._mod = _Oscillator(rate)

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        # Raised-cosine envelope: 1 - depth at the trough, 1 at the peak.
        mod = 1.0 - self.depth * 0.5 * (1.0 - self._mod(frames, sample_rate))
        mono = self._envelope(frames) * mod * self._tone(frames, sample_rate)
        return np.column_stack([mono, mono])


class AudioEngine:
    """Mixes voices into one stereo output stream. start/stop are reference counted."""

    def __init__(self, sample_rate: int = 44100, master: float = 1.0):
        self.sample_rate = sample_rate
        self.master = master
        self._voices: list[Voice] = []
        self._lock = threading.Lock()
        self._stream = None
        self._users = 0

    def add(self, voice: Voice) -> Voice:
        with self._lock:
            self._voices.append(voice)
        return voice

    def remove(self, voice: Voice) -> None:
        with self._lock:
            if voice in self._voices:
                self._voices.remove(voice)

    def render(self, frames: int) -> np.ndarray:
        with self._lock:
            voices = list(self._voices)
        out = np.zeros((frames, 2))
        for v in voices:
            out += v.render(frames, self.sample_rate)
        return np.clip(self.master * out, -1.0, 1.0)

    def start(self) -> None:
        self._users += 1
        if self._stream is not None:
            return
        try:
            import sounddevice
        except ImportError as e:
            self._users -= 1
            raise RuntimeError("audio needs sounddevice: pip install 'nova[audio]'") from e
        except OSError as e:  # sounddevice installed but the PortAudio system library is missing
            self._users -= 1
            raise RuntimeError(f"audio unavailable ({e}); on Linux: sudo apt install libportaudio2") from e
        try:
            self._stream = sounddevice.OutputStream(
                samplerate=self.sample_rate, channels=2, dtype="float32", callback=self._callback
            )
            self._stream.start()
        except Exception as e:
            self._users -= 1
            self._stream = None
            raise RuntimeError(f"could not open audio output: {e}") from e

    def stop(self) -> None:
        self._users = max(0, self._users - 1)
        if self._users == 0 and self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, outdata, frames, _time, _status) -> None:
        outdata[:] = self.render(frames).astype(np.float32)
