import json
import threading
import time
import urllib.request

from nova.devices import SimulatedDevice
from nova.live import DashboardFeedback, DashboardServer, dashboard_html
from nova.protocols import get_protocol
from nova.session import Session, SessionConfig


def test_dashboard_html_is_full_document():
    html = dashboard_html()
    assert html.startswith("<!doctype html>")
    assert "<title>NOVA</title>" in html
    assert "EventSource" in html


def read_events(url, stop_after, out, ready):
    with urllib.request.urlopen(url, timeout=10) as resp:
        ready.set()
        event, buf = None, []
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                buf.append(line[6:])
            elif line == "" and event:
                out.append((event, json.loads("".join(buf))))
                event, buf = None, []
                if out[-1][0] in stop_after:
                    return


def test_server_serves_page_and_streams_session():
    server = DashboardServer(port=0)
    sink = DashboardFeedback(open_browser=False, server=server)
    session = Session(
        SimulatedDevice(realtime=False, chunk_size=25, seed=1),
        get_protocol("alpha"),
        SessionConfig(baseline_sec=3, training_sec=3),
        [_Forward(sink)],
    )
    server.start()
    sink.configure(session)
    sink.open()
    events, ready = [], threading.Event()
    try:
        assert "NOVA" in urllib.request.urlopen(server.url, timeout=5).read().decode()

        reader = threading.Thread(target=read_events, args=(server.url + "stream", {"end"}, events, ready))
        reader.start()
        assert ready.wait(5)
        for _ in range(250):  # wait until the reader is registered as a client
            if server.client_count:
                break
            time.sleep(0.02)
        session.run()
    finally:
        sink.close()
        reader.join(10)
        server.stop()

    kinds = [k for k, _ in events]
    assert kinds[0] == "meta"
    assert events[0][1]["channels"] == ["SIM1", "SIM2", "SIM3", "SIM4"]
    updates = [d for k, d in events if k == "update"]
    assert updates, kinds
    assert all(len(u["samples"]) == 4 for u in updates)
    assert sum(len(u["samples"][0]) for u in updates) > 0
    assert "summary" in kinds
    assert kinds[-1] == "end"


class _Forward:
    """Passes updates to another sink without opening/closing it."""

    def __init__(self, sink):
        self.sink = sink

    def configure(self, session):
        pass

    def open(self):
        pass

    def update(self, state):
        self.sink.update(state)

    def finish(self, summary):
        self.sink.finish(summary)

    def close(self):
        pass


def test_home_screen_app_assets():
    server = DashboardServer(port=0)
    server.start()
    try:
        html = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert 'apple-mobile-web-app-capable" content="yes"' in html and 'rel="manifest"' in html
        manifest = json.loads(urllib.request.urlopen(server.url + "manifest.webmanifest", timeout=5).read())
        assert manifest["display"] == "fullscreen" and manifest["orientation"] == "landscape"
        icon = urllib.request.urlopen(server.url + "icon-180.png", timeout=5).read()
        assert icon[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        server.stop()


def test_static_site_build(tmp_path):
    import json

    from nova.site import build

    out = build(tmp_path / "site")
    html = (out / "index.html").read_text()
    assert 'href="manifest.webmanifest"' in html and 'register("sw.js")' in html
    assert 'href="/manifest.webmanifest"' not in html
    manifest = json.loads((out / "manifest.webmanifest").read_text())
    assert manifest["start_url"] == "./" and manifest["display"] == "fullscreen"
    assert all(not i["src"].startswith("/") for i in manifest["icons"])
    for name in ("sw.js", "icon-180.png", "icon-512.png", ".nojekyll"):
        assert (out / name).exists()
