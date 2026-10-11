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

    kinds = [k for k, _ in events]
    assert kinds[0] == "meta"
    assert events[0][1]["channels"] == ["SIM1", "SIM2", "SIM3", "SIM4"]
    updates = [d for k, d in events if k == "update"]
    assert updates, kinds
    assert all(len(u["samples"]) == 4 for u in updates)
    assert sum(len(u["samples"][0]) for u in updates) > 0
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

    def close(self):
        pass
