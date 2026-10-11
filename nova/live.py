"""NOVA app server: serves the session console and runs sessions behind it.

Standard library only. The console is served at http://127.0.0.1:<port>/,
receives live data over Server-Sent Events at /stream, and controls sessions
through a small JSON API:

  GET  /api/status           server mode and session state
  GET  /api/programs         built-in and saved programs, soundscapes
  POST /api/programs         save a custom program
  GET  /api/devices          headsets the app can connect to
  POST /api/session/start    {program, length, device: {...}, channels, line_freq}
  POST /api/session/skip     move to the next phase
  POST /api/session/stop     end the session early
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

import numpy as np

from .devices import COMMON_HEADSETS, brainflow_board_names, open_device_spec
from .feedback import FeedbackSink, FeedbackState

_SKELETON = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    "</head><body>{body}</body></html>"
)


def dashboard_html() -> str:
    """The console page, wrapped in a full HTML document."""
    body = files("nova").joinpath("dashboard/index.html").read_text(encoding="utf-8")
    return _SKELETON.format(body=body)


class DashboardServer:
    """HTTP server with a broadcast Server-Sent Events stream and an optional session API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, controller: "SessionController | None" = None):
        self.host = host
        self.port = port
        self.controller = controller
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._meta: str | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        server = self
        page = dashboard_html().encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the terminal clean
                pass

            def _json(self, code: int, payload) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif self.path == "/stream":
                    server._serve_stream(self)
                elif self.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                elif self.path == "/api/status":
                    ctl = server.controller
                    self._json(200, {"app": ctl is not None, "session": ctl.status() if ctl else None})
                elif self.path.startswith("/api/") and server.controller:
                    code, payload = server.controller.handle_get(self.path)
                    self._json(code, payload)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                if not (self.path.startswith("/api/") and server.controller):
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._json(400, {"error": "request body is not valid JSON"})
                    return
                code, payload = server.controller.handle_post(self.path, body)
                self._json(code, payload)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # resolves port 0 to the real one
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            for q in self._clients:
                q.put(None)
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def set_meta(self, meta: dict | None) -> None:
        self._meta = json.dumps(meta) if meta else None
        if meta:
            self.publish("meta", meta)

    def publish(self, event: str, payload: dict) -> None:
        message = f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(message)
            except queue.Full:  # a stalled browser tab must not stall the session
                pass

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _serve_stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        q: queue.Queue = queue.Queue(maxsize=400)
        if self._meta:
            q.put(f"event: meta\ndata: {self._meta}\n\n")
        with self._lock:
            self._clients.add(q)
        try:
            while True:
                try:
                    message = q.get(timeout=15)
                except queue.Empty:
                    message = ": keep-alive\n\n"
                if message is None:
                    break
                handler.wfile.write(message.encode("utf-8"))
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self._lock:
                self._clients.discard(q)


class DashboardFeedback(FeedbackSink):
    """Streams a running session to the NOVA console in the browser."""

    def __init__(self, port: int = 8765, open_browser: bool = True, server: DashboardServer | None = None):
        self._owns_server = server is None
        self.server = server or DashboardServer(port=port)
        self.open_browser = open_browser
        self._meta: dict = {}

    def configure(self, session) -> None:
        self._meta = {
            "device": session.device.description,
            "channels": session.channel_names,
            "sampling_rate": session.device.sampling_rate,
            "protocol": session.protocol.name,
            "metric_label": session.protocol.metric_label,
            "direction": session.protocol.direction,
            "program": session.program.to_dict(),
        }

    def open(self) -> None:
        if self._owns_server:
            self.server.start()
            print(f"NOVA console: {self.server.url}")
            if self.open_browser:
                webbrowser.open(self.server.url)
        self.server.set_meta(self._meta)

    def update(self, s: FeedbackState) -> None:
        samples = s.samples if s.samples is not None else np.empty((len(self._meta.get("channels", [])), 0))
        self.server.publish(
            "update",
            {
                "t": round(s.t, 3),
                "phase": s.phase,
                "phase_name": s.phase_name,
                "phase_index": s.phase_index,
                "phase_count": s.phase_count,
                "phase_elapsed": round(s.phase_elapsed, 2),
                "phase_duration": s.phase_duration,
                "session_duration": s.session_duration,
                "guidance": s.guidance,
                "binaural": s.binaural,
                "protocol": s.protocol,
                "metric_label": s.metric_label,
                "direction": s.direction,
                "metric": _finite(s.metric),
                "threshold": _finite(s.threshold),
                "reward": s.reward,
                "intensity": round(s.intensity, 3),
                "artifact": s.artifact,
                "inhibited": s.inhibited,
                "reward_fraction": round(s.reward_fraction, 4),
                "stim_on": s.stim_on,
                "state": s.brain_state,
                "target_rate": round(s.target_rate, 3),
                "samples": np.round(samples, 2).tolist(),
            },
        )

    def finish(self, summary) -> None:
        self.server.publish("summary", summary.to_dict())

    def close(self) -> None:
        self.server.publish("end", {})
        self.server.set_meta(None)
        if self._owns_server:
            self.server.stop()


class SessionController:
    """Runs one session at a time on behalf of the console."""

    def __init__(self, log_dir: str | Path = "sessions", programs_dir: Path | None = None):
        from .programs import CUSTOM_DIR

        self.log_dir = Path(log_dir)
        self.programs_dir = programs_dir or CUSTOM_DIR
        self.server: DashboardServer | None = None
        self.session = None
        self._state = {"state": "idle"}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def attach(self, server: DashboardServer) -> None:
        self.server = server
        server.controller = self

    def status(self) -> dict:
        return dict(self._state)

    def _set_state(self, **state) -> None:
        self._state = state
        if self.server:
            self.server.publish("status", state)

    # -- HTTP dispatch -------------------------------------------------------

    def handle_get(self, path: str) -> tuple[int, dict]:
        from .programs import SOUNDSCAPES, all_programs

        if path == "/api/programs":
            return 200, {
                "programs": [p.to_dict() for p in all_programs(self.programs_dir)],
                "soundscapes": SOUNDSCAPES,
            }
        if path == "/api/history":
            return 200, {"sessions": load_history(self.log_dir)}
        if path == "/api/devices":
            common = [{"id": i, "label": label, "connect": how} for i, label, how in COMMON_HEADSETS]
            known = {d["id"] for d in common}
            others = [
                {"id": n.removesuffix("_board"), "label": n.removesuffix("_board"), "connect": "serial"}
                for n in brainflow_board_names()
                if n.removesuffix("_board") not in known and "synthetic" not in n and "playback" not in n
            ]
            return 200, {"devices": common, "other": others}
        return 404, {"error": "not found"}

    def handle_post(self, path: str, body: dict) -> tuple[int, dict]:
        try:
            if path == "/api/session/start":
                return 200, self.start(body)
            if path == "/api/session/skip":
                if self.session:
                    self.session.skip_phase()
                return 200, self.status()
            if path == "/api/session/target":
                if not self.session:
                    raise ValueError("no session is running")
                self.session.set_reward_rate(float(body["reward_rate"]))
                return 200, {"reward_rate": self.session.reward_rate}
            if path == "/api/session/stop":
                if self.session:
                    self.session.stop()
                return 200, self.status()
            if path == "/api/programs":
                return 200, self.save_program(body)
        except (ValueError, KeyError, TypeError) as e:
            return 400, {"error": str(e)}
        return 404, {"error": "not found"}

    # -- actions -------------------------------------------------------------

    def save_program(self, body: dict) -> dict:
        from .programs import build_custom, save_custom

        protocol = {
            "reward": body["reward"],
            "ratio_over": body.get("ratio_over") or None,
            "direction": body.get("direction", "up"),
            "sites": [s.strip() for s in body.get("sites", []) if s.strip()],
        }
        program = build_custom(
            body.get("name") or "Custom program",
            protocol,
            minutes=float(body.get("minutes", 20)),
            blocks=int(body.get("blocks", 3)),
            reward_rate=float(body.get("reward_rate", 0.6)),
            soundscape=body.get("soundscape", "aurora"),
            eyes=body.get("eyes", "open"),
            binaural=float(body["binaural"]) if body.get("binaural") else None,
        )
        save_custom(program, self.programs_dir)
        return program.to_dict()

    def start(self, body: dict) -> dict:
        from .programs import get_program

        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("a session is already running")
            program = get_program(body["program"], self.programs_dir)
            length = float(body.get("length", 1.0))
            if length != 1.0:
                program = program.scaled(length)
            self._thread = threading.Thread(target=self._run, args=(program, body), daemon=True)
            self._set_state(state="connecting", program=program.id)
            self._thread.start()
        return self.status()

    def _run(self, program, body: dict) -> None:
        from .session import Session, SessionConfig

        device_spec = dict(body.get("device") or {"id": "demo"})
        line = float(body.get("line_freq") or 60.0)
        device_spec.setdefault("line_freq", line)
        try:
            device = open_device_spec(device_spec)
            config = SessionConfig(notch_hz=line or None, channels=body.get("channels") or None)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = self.log_dir / f"{stamp}-{program.id}.csv"
            sink = DashboardFeedback(server=self.server, open_browser=False)
            self.session = Session(device, config=config, sinks=[sink], log_path=log_path, program=program)
            self._set_state(state="running", program=program.id, device=device.description)
            summary = self.session.run()
            self._set_state(
                state="finished", program=program.id, stopped_early=summary.aborted, log=str(log_path)
            )
        except Exception as e:  # report any connection or session failure to the console
            traceback.print_exc()
            self._set_state(state="error", program=program.id, error=str(e))
        finally:
            self.session = None


def load_history(log_dir: str | Path, limit: int = 300) -> list[dict]:
    """Saved session summaries, newest first."""
    out = []
    for path in sorted(Path(log_dir).glob("*.json"), reverse=True)[:limit]:
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _finite(x: float) -> float | None:
    return round(float(x), 6) if np.isfinite(x) else None
