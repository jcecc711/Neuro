"""EEG acquisition devices.

Every device yields EEG in microvolts as an array shaped (channels, samples).
`read()` returns only the samples that arrived since the previous call, so the
session loop can stay agnostic about how the hardware buffers data.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import lfilter


class DeviceError(RuntimeError):
    pass


class Device(ABC):
    sampling_rate: float
    channel_names: list[str]

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return new EEG samples (µV) shaped (channels, samples); may be empty."""

    @property
    def description(self) -> str:
        return type(self).__name__

    def __enter__(self) -> "Device":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class SimulatedDevice(Device):
    """Synthetic EEG: pink background, slowly waxing/waning alpha, line noise.

    With ``realtime=True`` samples are produced at wall-clock pace like real
    hardware. With ``realtime=False`` every ``read()`` returns ``chunk_size``
    samples immediately, which lets tests run a full session in milliseconds.
    """

    # Pink-noise (1/f) shaping filter, Paul Kellet's coefficients.
    _PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    _PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

    def __init__(
        self,
        n_channels: int = 4,
        sampling_rate: float = 250.0,
        line_freq: float = 60.0,
        realtime: bool = True,
        chunk_size: int = 25,
        seed: int | None = None,
    ):
        self.sampling_rate = float(sampling_rate)
        self.channel_names = [f"SIM{i + 1}" for i in range(n_channels)]
        self.line_freq = line_freq
        self.realtime = realtime
        self.chunk_size = chunk_size
        self._rng = np.random.default_rng(seed)
        self._zi = np.zeros((n_channels, len(self._PINK_A) - 1))
        self._n = 0
        self._t0: float | None = None

    @property
    def description(self) -> str:
        mode = "realtime" if self.realtime else "fast"
        return f"Simulator ({len(self.channel_names)} ch @ {self.sampling_rate:g} Hz, {mode})"

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._n = 0

    def stop(self) -> None:
        self._t0 = None

    def read(self) -> np.ndarray:
        if self._t0 is None:
            raise DeviceError("device not started")
        if self.realtime:
            due = int((time.monotonic() - self._t0) * self.sampling_rate)
            count = due - self._n
        else:
            count = self.chunk_size
        if count <= 0:
            return np.empty((len(self.channel_names), 0))
        return self._generate(count)

    def _generate(self, count: int) -> np.ndarray:
        fs = self.sampling_rate
        n_ch = len(self.channel_names)
        t = (self._n + np.arange(count)) / fs
        self._n += count

        white = self._rng.standard_normal((n_ch, count))
        pink, self._zi = lfilter(self._PINK_B, self._PINK_A, white, axis=1, zi=self._zi)
        signal = 15.0 * pink

        # Alpha that drifts up and down over ~20 s so feedback has something to track.
        alpha_amp = 9.0 + 7.0 * np.sin(2 * np.pi * t / 20.0)
        alpha = alpha_amp * np.sin(2 * np.pi * 10.0 * t)
        beta = 2.0 * np.sin(2 * np.pi * 20.0 * t + 1.0)
        line = 4.0 * np.sin(2 * np.pi * self.line_freq * t)
        # Posterior channels get more alpha, like a real montage.
        weights = np.linspace(0.6, 1.0, n_ch)[:, None]
        return signal + weights * alpha + beta + line


class BrainFlowDevice(Device):
    """Any board supported by BrainFlow (OpenBCI, Muse, Neurosity, BrainBit, ...)."""

    def __init__(
        self,
        board: str | int,
        serial_port: str = "",
        mac_address: str = "",
        ip_address: str = "",
        ip_port: int = 0,
        serial_number: str = "",
        other_info: str = "",
        timeout: int = 0,
        verbose: bool = False,
    ):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
        except ImportError as e:  # pragma: no cover - depends on environment
            raise DeviceError("BrainFlow is not installed: pip install 'neuro[brainflow]'") from e

        self._BoardShim = BoardShim
        self.board_id = resolve_brainflow_board(board)
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        params.ip_address = ip_address
        params.ip_port = ip_port
        params.serial_number = serial_number
        params.other_info = other_info
        params.timeout = timeout

        if verbose:
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        self._board = BoardShim(self.board_id, params)
        self.sampling_rate = float(BoardShim.get_sampling_rate(self.board_id))
        self._eeg_rows = BoardShim.get_eeg_channels(self.board_id)
        try:
            self.channel_names = list(BoardShim.get_eeg_names(self.board_id))
        except Exception:
            self.channel_names = [f"EEG{i + 1}" for i in range(len(self._eeg_rows))]
        self._running = False

    @property
    def description(self) -> str:
        from brainflow.board_shim import BoardIds

        name = BoardIds(self.board_id).name
        return f"BrainFlow {name} ({len(self.channel_names)} ch @ {self.sampling_rate:g} Hz)"

    def start(self) -> None:
        try:
            self._board.prepare_session()
            self._board.start_stream()
        except Exception as e:
            raise DeviceError(f"could not start board: {e}") from e
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._board.stop_stream()
        finally:
            self._board.release_session()

    def read(self) -> np.ndarray:
        data = self._board.get_board_data()  # drains the internal ring buffer
        return np.asarray(data[self._eeg_rows], dtype=float)


class LSLDevice(Device):
    """Any EEG source publishing a Lab Streaming Layer stream."""

    def __init__(self, stream_name: str | None = None, stream_type: str = "EEG", timeout: float = 10.0):
        try:
            import pylsl
        except ImportError as e:  # pragma: no cover - depends on environment
            raise DeviceError("pylsl is not installed: pip install 'neuro[lsl]'") from e

        prop, value = ("name", stream_name) if stream_name else ("type", stream_type)
        streams = pylsl.resolve_byprop(prop, value, timeout=timeout)
        if not streams:
            raise DeviceError(f"no LSL stream with {prop}={value!r} found within {timeout:g}s")
        info = streams[0]
        self._inlet = pylsl.StreamInlet(info, max_buflen=60)
        self.sampling_rate = float(info.nominal_srate())
        if self.sampling_rate <= 0:
            raise DeviceError("LSL stream has an irregular sampling rate; neurofeedback needs a fixed rate")
        self._name = info.name()
        self.channel_names = _lsl_channel_labels(info) or [f"CH{i + 1}" for i in range(info.channel_count())]

    @property
    def description(self) -> str:
        return f"LSL '{self._name}' ({len(self.channel_names)} ch @ {self.sampling_rate:g} Hz)"

    def start(self) -> None:
        self._inlet.open_stream()

    def stop(self) -> None:
        self._inlet.close_stream()

    def read(self) -> np.ndarray:
        samples, _ = self._inlet.pull_chunk(timeout=0.0)
        if not samples:
            return np.empty((len(self.channel_names), 0))
        return np.asarray(samples, dtype=float).T


def _lsl_channel_labels(info) -> list[str]:
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels if len(labels) == info.channel_count() and all(labels) else []


def brainflow_board_names() -> list[str]:
    try:
        from brainflow.board_shim import BoardIds
    except ImportError:
        return []
    return sorted(b.name.lower() for b in BoardIds if b.value >= -1)


def resolve_brainflow_board(board: str | int) -> int:
    """Accept a BoardIds value, a full name ('CYTON_BOARD') or a short one ('cyton')."""
    from brainflow.board_shim import BoardIds

    if isinstance(board, int) or str(board).lstrip("-").isdigit():
        return BoardIds(int(board)).value
    key = str(board).upper().replace("-", "_")
    for candidate in (key, f"{key}_BOARD"):
        if candidate in BoardIds.__members__:
            return BoardIds[candidate].value
    raise DeviceError(f"unknown BrainFlow board {board!r}; run 'neuro devices' to list them")
