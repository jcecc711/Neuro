"""Command-line interface: `nova devices | protocols | check | run`."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .devices import (
    DEMO_MONTAGE,
    BrainFlowDevice,
    Device,
    DeviceError,
    LSLDevice,
    SimulatedDevice,
    brainflow_board_names,
)
from .programs import all_programs, get_program
from .dsp import StreamingFilter, band_power, power_spectrum
from .audio import AudioEngine
from .feedback import FEEDBACK_KINDS, make_sinks
from .protocols import BANDS, PROTOCOLS, get_protocol
from .session import Session, SessionConfig
from .stimulation import SCHEDULES, Stimulus, StimulusSpec


def add_device_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("device")
    g.add_argument(
        "--device",
        default="sim",
        help="'demo' (demo signal, no headset), 'lsl', or a BrainFlow board name such as "
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
    if name in ("sim", "simulator", "demo"):
        return SimulatedDevice(montage=DEMO_MONTAGE, line_freq=args.line_freq or 60.0)
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
    print("Built-in:\n  demo       demo signal, no headset required\n  lsl        any Lab Streaming Layer EEG stream")
    names = brainflow_board_names()
    if names:
        print("\nBrainFlow boards (use with --device):")
        for n in names:
            print(f"  {n.removesuffix('_board')}")
    else:
        print("\nBrainFlow not installed; pip install 'nova[brainflow]' for OpenBCI, Muse, etc.")
    return 0


def cmd_protocols(_args: argparse.Namespace) -> int:
    for p in PROTOCOLS.values():
        direction = "up" if p.direction > 0 else "down"
        bands = ", ".join(f"{b} {BANDS[b][0]:g}-{BANDS[b][1]:g} Hz" for b in p.inhibits)
        print(f"{p.name}\n  {p.description}")
        print(f"  trains {p.metric_label} {direction}; inhibits: {bands}")
        print(f"  suggested sites: {', '.join(p.suggested_sites)}\n")
    return 0


def cmd_programs(_args: argparse.Namespace) -> int:
    for p in all_programs():
        badge = f"  [{p.evidence}]" if p.evidence else ""
        print(f"{p.id}  {p.name}: {p.target}  ({p.minutes:g} min, {p.soundscape}, eyes {p.eyes}){badge}")
        print(f"  {p.summary}")
        for ph in p.phases:
            proto = f"  {ph.resolved_protocol().metric_label}" if ph.protocol else ""
            print(f"    {ph.minutes:4.1f} min  {ph.name:<24}{proto}")
        print()
    print("NOVA is not a medical device. Programs follow research protocols; they don't treat conditions.")
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


def add_stim_args(p: argparse.ArgumentParser, required: bool = False) -> None:
    g = p.add_argument_group("auditory stimulation (use headphones)")
    g.add_argument(
        "--stim",
        choices=["binaural", "pulsed"],
        required=required,
        default=None,
        help="binaural: beat between ears (default 10 Hz); pulsed: amplitude-modulated tone (default 40 Hz)",
    )
    g.add_argument("--stim-freq", type=float, default=None, help="beat / pulse frequency in Hz")
    g.add_argument("--stim-carrier", type=float, default=None, help="carrier tone in Hz (200 binaural, 500 pulsed)")
    g.add_argument("--stim-volume", type=float, default=0.15, help="0-1, default 0.15. Start low")


def stim_spec(args: argparse.Namespace, schedule: str = "always", block_sec: float = 30.0) -> StimulusSpec:
    return StimulusSpec.create(args.stim, args.stim_freq, args.stim_carrier, args.stim_volume, schedule, block_sec)


def cmd_stim(args: argparse.Namespace) -> int:
    """Play stimulation on its own, without EEG, e.g. to try it on headphones."""
    spec = stim_spec(args)
    engine = AudioEngine()
    stim = Stimulus(spec, engine)
    if args.out:
        write_stim_wav(stim, args.seconds, args.out)
        print(f"Wrote {args.seconds:g}s of {spec.sound} to {args.out}")
        return 0
    print(f"Playing {spec.sound} for {args.seconds:g}s. Ctrl-C to stop.")
    stim.open()
    try:
        stim.step(0.0, 0.0)
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        stim.close()
    return 0


def write_stim_wav(stim: Stimulus, seconds: float, path: str, fade_sec: float = 2.0) -> None:
    from scipy.io import wavfile

    engine = stim.engine
    engine.add(stim.voice)
    stim.step(0.0, 0.0)
    audio = engine.render(int(seconds * engine.sample_rate))
    engine.remove(stim.voice)
    fade = min(int(fade_sec * engine.sample_rate), len(audio) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    wavfile.write(path, engine.sample_rate, (audio * 32767).astype(np.int16))


def cmd_run(args: argparse.Namespace) -> int:
    program = None
    if args.program:
        program = get_program(args.program)
        program = program.fitted(args.minutes) if args.minutes else program.scaled(args.length)
    protocol = get_protocol(args.protocol)
    spec = stim_spec(args, args.stim_schedule, args.stim_block) if args.stim else None
    # Keep the stimulation frequency inside the analysed band so its EEG response is visible.
    high_hz = max(40.0, spec.freq + 5.0) if spec else 40.0
    config = SessionConfig(
        baseline_sec=args.baseline,
        training_sec=args.duration,
        target_reward_rate=args.reward_rate,
        freeze_thresholds=args.freeze_thresholds,
        sustain_sec=args.sustain,
        artifact_uv=args.artifact_uv,
        notch_hz=args.line_freq or None,
        channels=args.channels.split(",") if args.channels else None,
        high_hz=high_hz,
    )
    log_path = None
    if not args.no_log:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = Path(args.log_dir) / f"{stamp}-{program.id if program else protocol.name}.csv"

    device = open_device(args)
    engine = AudioEngine()
    sinks = make_sinks(args.feedback.split(","), engine, dashboard_port=args.dashboard_port)
    stimulus = Stimulus(spec, engine) if spec else None
    session = Session(device, protocol, config, sinks, log_path, stimulus, program=program)
    print(f"{device.description}")
    if program:
        print(f"Program '{program.name}' ({program.minutes:g} min): {program.summary}")
    else:
        print(f"Protocol '{protocol.name}': {protocol.description}")
    print(f"Channels: {', '.join(session.channel_names)}")
    if spec:
        print(f"Stimulus: {spec.description}")
    if not program:
        print(f"Baseline {config.baseline_sec:g}s, then training {config.training_sec:g}s.")
    print("Ctrl-C to stop.\n")
    summary = session.run()
    print("\n" + summary.format())
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """Run the NOVA app: the session console in the browser, backed by this process."""
    from .live import DashboardServer, SessionController

    controller = SessionController(log_dir=args.log_dir)
    server = DashboardServer(host=args.host, port=args.port)
    controller.attach(server)
    server.start()
    if args.host in ("0.0.0.0", "::"):
        ip = _lan_ip()
        print(f"NOVA is running. On this computer: http://127.0.0.1:{server.port}/")
        print(f"On your iPad or phone (same Wi-Fi): http://{ip}:{server.port}/")
        print("Anyone on this network can open it and start sessions. Ctrl-C to quit.")
        server.host = "127.0.0.1"
    else:
        print(f"NOVA is running at {server.url}  Ctrl-C to quit.")
    if not args.no_browser:
        import webbrowser

        webbrowser.open(server.url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if controller.session:
            controller.session.stop()
    finally:
        server.stop()
    return 0


def _lan_ip() -> str:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packets are sent; this picks the outbound interface
            return s.getsockname()[0]
        except OSError:
            return "your-computer-ip"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nova", description="NOVA: real-time EEG neurofeedback")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list supported hardware").set_defaults(func=cmd_devices)
    sub.add_parser("protocols", help="list training protocols").set_defaults(func=cmd_protocols)

    check = sub.add_parser("check", help="check signal quality / electrode contact")
    add_device_args(check)
    check.add_argument("--seconds", type=float, default=60.0)
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="run a neurofeedback session")
    add_device_args(run)
    run.add_argument("--program", default=None, help="program id or .json file; overrides --protocol/--baseline/--duration")
    run.add_argument("--minutes", type=float, default=None, help="fit the program to this many minutes (15-60)")
    run.add_argument("--length", type=float, default=1.0, help="scale training blocks, e.g. 0.5 or 1.5")
    run.add_argument("--protocol", default="alpha", choices=list(PROTOCOLS))
    run.add_argument("--channels", default=None, help="comma-separated channel names (default: all)")
    run.add_argument("--baseline", type=float, default=60.0, help="calibration seconds (default 60)")
    run.add_argument("--duration", type=float, default=300.0, help="training seconds (default 300)")
    run.add_argument("--reward-rate", type=float, default=0.6, help="target fraction of time rewarded")
    run.add_argument("--sustain", type=float, default=0.5, help="seconds criteria must hold before reward")
    run.add_argument("--freeze-thresholds", action="store_true", help="fix thresholds after baseline")
    run.add_argument("--artifact-uv", type=float, default=200.0, help="peak-to-peak artifact limit in µV")
    run.add_argument(
        "--feedback", default="console", help=f"comma list of: {', '.join(FEEDBACK_KINDS)} (default console)"
    )
    run.add_argument("--dashboard-port", type=int, default=8765, help="port for --feedback dashboard")
    run.add_argument("--log-dir", default="sessions")
    run.add_argument("--no-log", action="store_true")
    add_stim_args(run)
    run.add_argument(
        "--stim-schedule",
        choices=SCHEDULES,
        default="blocks",
        help="when the stimulus plays; 'blocks' alternates on/off so its effect can be measured (default)",
    )
    run.add_argument("--stim-block", type=float, default=30.0, help="seconds per on/off block (default 30)")
    run.set_defaults(func=cmd_run)

    stim = sub.add_parser("stim", help="play binaural beats or pulsed tones without EEG")
    add_stim_args(stim, required=True)
    stim.add_argument("--seconds", type=float, default=300.0)
    stim.add_argument("--out", default=None, help="write a .wav file instead of playing (e.g. to play on a phone)")
    stim.set_defaults(func=cmd_stim)

    app = sub.add_parser("app", aliases=["dashboard"], help="open the NOVA session console in your browser")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open NOVA from an iPad or phone on your Wi-Fi")
    app.add_argument("--no-browser", action="store_true")
    app.add_argument("--log-dir", default="sessions")
    app.set_defaults(func=cmd_app)

    sub.add_parser("programs", help="list session programs").set_defaults(func=cmd_programs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DeviceError, ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
