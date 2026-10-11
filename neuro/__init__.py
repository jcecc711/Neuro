"""Real-time EEG neurofeedback."""

from .devices import BrainFlowDevice, Device, DeviceError, LSLDevice, SimulatedDevice
from .protocols import PROTOCOLS, Protocol, get_protocol
from .session import Session, SessionConfig, SessionSummary

__all__ = [
    "BrainFlowDevice",
    "Device",
    "DeviceError",
    "LSLDevice",
    "SimulatedDevice",
    "PROTOCOLS",
    "Protocol",
    "get_protocol",
    "Session",
    "SessionConfig",
    "SessionSummary",
]
__version__ = "0.1.0"
