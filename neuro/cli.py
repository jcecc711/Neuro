"""Command-line interface: `neuro devices | protocols | check | run`."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .devices import BrainFlowDevice, Device, DeviceError, LSLDevice, SimulatedDevice, brainflow_board_names
from .dsp import StreamingFilter, band_power, power_spectrum
from .feedback import make_sinks
from .protocols import BANDS, PROTOCOLS, get_protocol
from .session import Session, SessionConfig


def add_device_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("device")
    g.add_argument(
        "--device",
        default="sim",
        help="'sim' (built-in simulator), 'lsl', or a BrainFlow board name such as "
        "cyton, ganglion, muse_2, synthetic (default: sim)",
    )
    g.add_argument("--serial-port", default="", help="e.g. /dev/ttyUSB0 or COM3 (OpenBCI dongle)")
    g.add_argument("--mac-address", default="", help="Bluetooth MAC address, if the board needs it")
    g.add_argument("--ip-address", default="", help="IP address for WiFi boards")
    g.add_argument("--ip-port", type=int, default=0)
    g.add_argument("--serial-number", default="", help="device serial/name, e.g. for Muse")
    g.add_argument("--other-info", default="", help="board-specific extra BrainFlow parameter")
    g.add_argument("--lsl-name", default=None, help="LSL stream name (default: first EEG stream)")
    g.add_argument("--line-freq", type=float, default=60.0, help="mains frequency to notch: 50 or 60 (0 = off)")
    g.add_argument("--verbose-device", action="store_true", help="show driver logs")


def open_device(args: argparse.Namespace) -> Device:
    name = args.device.lower()
    if name in ("sim", "simulator"):
        return SimulatedDevice(line_freq=args.line_freq or 60.0)
    if name == "lsl":
        return LSLDevice(stream_name=args.lsl_name)
    return BrainFlowDevice(
        name,
        serial_port=args.serial_port,
        mac_address=args.mac_address,
        ip_address=args.ip_address,
        ip_port=args.ip_port,
        serial_number=args.serial_number,
        other_info=args.other_info,
        verbose=args.verbose_device,
    )


def cmd_devices(_args: argparse.Namespace) -> int:
    print("Built-in:\n  sim        simulated EEG, no hardware required\n  lsl        any Lab Streaming Layer EEG stream")
    names = brainflow_board_names()
    if names:
        print("\nBrainFlow boards (use with --device):")
        for n in names:
            print(f"  {n.removesuffix('_board')}")
    else:
        print("\nBrainFlow not installed; pip install 'neuro[brainflow]' for OpenBCI, Muse, etc.")
    return 0


def cmd_protocols(_args: argparse.Namespace) -> int:
    for p in PROTOCOLS.values():
        direction = "up" if p.direction > 0 else "down"
        bands = ", ".join(f"{b} {BANDS[b][0]:g}-{BANDS[b][1]:g} Hz" for b in p.inhibits)
        print(f"{p.name}\n  {p.description}")
        print(f"  trains {p.metric_label} {direction}; inhibits: {bands}")
        print(f"  suggested sites: {', '.join(p.suggested_sites)}\n")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Live per-channel signal quality readout, for checking electrode contact."""
    device = open_device(args)
    fs = device.sampling_rate
    filt = StreamingFilter(fs, len(device.channel_names), notch=args.line_freq or None)
    raw_chunks: list[np.ndarray] = []
    print(f"{device.description}. Ctrl-C to stop.\n")
    header = f"{'channel':<10}{'RMS µV':>9}{'line µV²':>10}{'alpha µV²':>11}{'status':>10}"
    try:
        with device:
            end = time.monotonic() + args.seconds
            next_print = time.monotonic() + 1.0
            while time.monotonic() < end:
                chunk = device.read()
                if chunk.shape[1]:
                    raw_chunks.append(chunk)
                if time.monotonic() < next_print:
                    time.sleep(0.02)
                    continue
                next_print += 1.0
                raw = np.concatenate(raw_chunks, axis=1)[:, -int(2 * fs) :] if raw_chunks else None
                raw_chunks = [raw] if raw is not None else []
                if raw is None or raw.shape[1] < fs:
                    continue
                clean = filt.process(raw)
                freqs, psd_raw = power_spectrum(raw - raw.mean(axis=1, keepdims=True), fs)
                line = args.line_freq or 60.0
                line_p = band_power(freqs, psd_raw, (line - 2, line + 2))
                _, psd = power_spectrum(clean, fs)
                alpha = band_power(freqs, psd, BANDS["alpha"])
                rms = clean.std(axis=1)
                print(header)
                for i, ch in enumerate(device.channel_names):
                    status = "ok"
                    if rms[i] < 0.5:
                        status = "flat"
                    elif rms[i] > 100 or line_p[i] > 100:
                        status = "noisy"
                    print(f"{ch:<10}{rms[i]:>9.1f}{line_p[i]:>10.1f}{alpha[i]:>11.1f}{status:>10}")
                print()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    protocol = get_protocol(args.protocol)
    config = SessionConfig(
        baseline_sec=args.baseline,
        training_sec=args.duration,
        target_reward_rate=args.reward_rate,
        freeze_thresholds=args.freeze_thresholds,
        sustain_sec=args.sustain,
        artifact_uv=args.artifact_uv,
        notch_hz=args.line_freq or None,
        channels=args.channels.split(",") if args.channels else None,
    )
    log_path = None
    if not args.no_log:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = Path(args.log_dir) / f"{stamp}-{protocol.name}.csv"

    device = open_device(args)
    sinks = make_sinks(args.feedback.split(","))
    session = Session(device, protocol, config, sinks, log_path)
    print(f"{device.description}")
    print(f"Protocol '{protocol.name}': {protocol.description}")
    print(f"Channels: {', '.join(session.channel_names)}")
    print(f"Baseline {config.baseline_sec:g}s, then training {config.training_sec:g}s. Ctrl-C to stop.\n")
    summary = session.run()
    print("\n" + summary.format())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuro", description="Real-time EEG neurofeedback trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list supported hardware").set_defaults(func=cmd_devices)
    sub.add_parser("protocols", help="list training protocols").set_defaults(func=cmd_protocols)

    check = sub.add_parser("check", help="check signal quality / electrode contact")
    add_device_args(check)
    check.add_argument("--seconds", type=float, default=60.0)
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="run a neurofeedback session")
    add_device_args(run)
    run.add_argument("--protocol", default="alpha", choices=list(PROTOCOLS))
    run.add_argument("--channels", default=None, help="comma-separated channel names (default: all)")
    run.add_argument("--baseline", type=float, default=60.0, help="calibration seconds (default 60)")
    run.add_argument("--duration", type=float, default=300.0, help="training seconds (default 300)")
    run.add_argument("--reward-rate", type=float, default=0.6, help="target fraction of time rewarded")
    run.add_argument("--sustain", type=float, default=0.5, help="seconds criteria must hold before reward")
    run.add_argument("--freeze-thresholds", action="store_true", help="fix thresholds after baseline")
    run.add_argument("--artifact-uv", type=float, default=200.0, help="peak-to-peak artifact limit in µV")
    run.add_argument("--feedback", default="console", help="comma list of: console, audio, visual")
    run.add_argument("--log-dir", default="sessions")
    run.add_argument("--no-log", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DeviceError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
