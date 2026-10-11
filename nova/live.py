"""NOVA dashboard server: serves the web console and streams sessions into it.

Standard library only. The page is served at http://127.0.0.1:<port>/ and
receives data over Server-Sent Events at /stream. With no session running the
page falls back to its built-in simulation.
"""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

import numpy as np

from .feedback import FeedbackSink, FeedbackState

_SKELETON = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    "</head><body>{body}</body></html>"
)


def dashboard_html() -> str:
    """The dashboard page, wrapped in a full HTML document."""
    body = files("nova").joinpath("dashboard/index.html").read_text(encoding="utf-8")
    return _SKELETON.format(body=body)


class DashboardServer:
    """HTTP server with a broadcast Server-Sent Events stream."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
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
                else:
                    self.send_error(404)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]  # resolves port 0 to the real one
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.publish("end", {})
        with self._lock:
            for q in self._clients:
                q.put(None)
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def set_meta(self, meta: dict) -> None:
        self._meta = json.dumps(meta)
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
        q: queue.Queue = queue.Queue(maxsize=200)
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
    """Streams a running session to the NOVA dashboard in the browser."""

    def __init__(self, port: int = 8765, open_browser: bool = True, server: DashboardServer | None = None):
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
        }

    def open(self) -> None:
        self.server.start()
        self.server.set_meta(self._meta)
        print(f"NOVA dashboard: {self.server.url}")
        if self.open_browser:
            webbrowser.open(self.server.url)

    def update(self, s: FeedbackState) -> None:
        samples = s.samples if s.samples is not None else np.empty((len(self._meta.get("channels", [])), 0))
        self.server.publish(
            "update",
            {
                "t": round(s.t, 3),
                "phase": s.phase,
                "metric": _finite(s.metric),
                "threshold": _finite(s.threshold),
                "reward": s.reward,
                "artifact": s.artifact,
                "inhibited": s.inhibited,
                "reward_fraction": round(s.reward_fraction, 4),
                "stim_on": s.stim_on,
                "samples": np.round(samples, 2).tolist(),
            },
        )

    def close(self) -> None:
        self.server.stop()


def _finite(x: float) -> float | None:
    return round(float(x), 6) if np.isfinite(x) else None
