import json
import time
import urllib.request

import numpy as np
import pytest

from nova.devices import DEMO_MONTAGE, SimulatedDevice
from nova.programs import (
    BUILTIN,
    KINDS,
    SOUNDSCAPES,
    Phase,
    Program,
    build_custom,
    export_json,
    get_program,
    load_custom,
    save_custom,
)
from nova.session import Session, SessionConfig


def demo_device(**kw):
    return SimulatedDevice(montage=DEMO_MONTAGE, realtime=False, chunk_size=25, seed=4, **kw)


def tiny(program: Program, seconds: float = 6.0) -> Program:
    """Shrink every phase so a whole program runs in a fraction of a second."""
    from dataclasses import replace

    return replace(program, phases=tuple(replace(p, minutes=seconds / 60) for p in program.phases))


@pytest.mark.parametrize("program", BUILTIN, ids=lambda p: p.id)
def test_builtin_programs_follow_the_arc(program):
    kinds = [p.kind for p in program.phases]
    assert kinds[0] == "baseline" and kinds[-1] == "cooldown"
    assert program.phases[1].kind == "training" and program.phases[1].reward_rate > 0.7  # easy warm-up
    assert "rest" in kinds  # breathing interludes between blocks
    assert all(k in KINDS for k in kinds)
    assert program.soundscape in SOUNDSCAPES
    assert program.evidence in ("Most studied", "Emerging", "Limited")
    assert program.evidence_note
    assert 15 <= program.minutes <= 35


@pytest.mark.parametrize("program", BUILTIN, ids=lambda p: p.id)
def test_builtin_programs_run_on_demo_montage(program):
    rec = []
    session = Session(demo_device(), config=SessionConfig(), program=tiny(program), sinks=[_Rec(rec)])
    summary = session.run()
    assert not summary.aborted
    assert [p.name for p in summary.phases] == [p.name for p in program.phases]
    assert {s.phase for s in rec} == {p.kind for p in program.phases}
    assert not any(s.reward for s in rec if s.phase != "training")
    assert all(np.isfinite(s.metric) for s in rec)


def test_conditions_cover_requested_areas():
    targets = " ".join(p.target.lower() for p in BUILTIN if p.category == "condition")
    for word in ("adhd", "depression", "anxiety", "stress", "fatigue"):
        assert word in targets


def test_depression_program_carries_crisis_note():
    assert "988" in get_program("mood-depression").caution


def test_asymmetry_program_rejects_headset_without_pair():
    with pytest.raises(ValueError, match="left/right"):
        Session(SimulatedDevice(realtime=False), program=get_program("mood-depression"))


def test_phase_validation():
    with pytest.raises(ValueError):
        Phase("x", "training", 1)  # training needs a protocol
    with pytest.raises(ValueError):
        Phase("x", "nap", 1)
    with pytest.raises(ValueError):
        Program("p", "P", "custom", "", "", (Phase("a", "baseline", 1),))  # no training phase


def test_scaled_changes_only_training():
    p = get_program("focus-adhd")
    s = p.scaled(0.5)
    for a, b in zip(p.phases, s.phases):
        assert b.minutes == (round(a.minutes * 0.5, 2) if a.kind == "training" else a.minutes)


def test_custom_program_roundtrip(tmp_path):
    p = build_custom("Evening SMR", {"reward": "smr", "direction": "up", "sites": ["C4"]}, minutes=20, blocks=2,
                     soundscape="ocean", binaural=14)
    assert p.id == "custom-evening-smr"
    assert p.minutes == pytest.approx(20, abs=0.1)
    assert sum(ph.kind == "training" for ph in p.phases) == 3  # warm-up + 2 blocks
    save_custom(p, tmp_path)
    loaded = load_custom(tmp_path)
    assert [x.id for x in loaded] == [p.id]
    assert loaded[0].phases == p.phases
    assert get_program("custom-evening-smr", tmp_path).name == "Evening SMR"
    (tmp_path / "broken.json").write_text("{not json")
    assert len(load_custom(tmp_path)) == 1  # bad files are skipped


def test_skip_and_stop():
    session = Session(demo_device(), program=tiny(get_program("calm-anxiety"), seconds=30))
    session.skip_phase()
    with session.device:
        for _ in range(50):
            session.process_chunk(session.device.read())
    assert session.phase.name == "Warm-up"
    session.stop()
    summary = session.run()
    assert summary.aborted


def test_warmup_is_easier_than_blocks():
    rec = []
    session = Session(demo_device(), program=tiny(get_program("stress-recovery"), seconds=40), sinks=[_Rec(rec)])
    session.run()
    by_phase = {}
    for s in rec:
        if s.phase == "training":
            by_phase.setdefault(s.phase_name, []).append(s.reward)
    assert np.mean(by_phase["Warm-up"]) > np.mean(by_phase["Rest in it"])


def test_dashboard_embeds_current_library():
    from importlib.resources import files

    html = files("nova").joinpath("dashboard/index.html").read_text()
    start = html.index('id="nova-programs">') + len('id="nova-programs">')
    embedded = json.loads(html[start : html.index("</script>", start)])
    assert embedded == json.loads(export_json()), "run: python -m nova.programs"


def test_app_api_runs_a_session(tmp_path):
    from nova.live import DashboardServer, SessionController

    ctl = SessionController(log_dir=tmp_path / "logs", programs_dir=tmp_path / "programs")
    server = DashboardServer(port=0)
    ctl.attach(server)
    server.start()
    try:
        def call(path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(server.url.rstrip("/") + path, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())

        assert call("/api/status")["app"] is True
        programs = call("/api/programs")
        assert {p["id"] for p in programs["programs"]} >= {"focus-adhd", "mood-depression"}
        assert any(d["id"] == "demo" for d in call("/api/devices")["devices"])

        saved = call("/api/programs", {"name": "Quick alpha", "reward": "alpha", "sites": ["Pz"], "minutes": 10, "blocks": 1})
        assert saved["id"] == "custom-quick-alpha"

        call("/api/session/start", {"program": "custom-quick-alpha", "device": {"id": "demo"}})
        for _ in range(100):
            if call("/api/status")["session"]["state"] == "running":
                break
            time.sleep(0.05)
        assert call("/api/status")["session"]["state"] == "running"
        call("/api/session/skip", {})
        call("/api/session/stop", {})
        for _ in range(100):
            if call("/api/status")["session"]["state"] == "finished":
                break
            time.sleep(0.05)
        status = call("/api/status")["session"]
        assert status["state"] == "finished" and status["stopped_early"]
        assert list((tmp_path / "logs").glob("*.csv"))
    finally:
        server.stop()


class _Rec:
    def __init__(self, out):
        self.out = out

    def configure(self, session):
        pass

    def open(self):
        pass

    def update(self, state):
        self.out.append(state)

    def finish(self, summary):
        pass

    def close(self):
        pass


def test_reward_rate_can_change_mid_session():
    session = Session(demo_device(), program=tiny(get_program("stress-recovery"), seconds=20))
    assert session.reward_rate == 0.6
    with pytest.raises(ValueError):
        session.set_reward_rate(1.5)
    session.set_reward_rate(0.5)
    with session.device:
        for _ in range(int(25 * 250 / 25)):  # into the warm-up
            session.process_chunk(session.device.read())
    assert session.phase.name == "Warm-up"
    assert session.reward_rate == pytest.approx(0.65)  # warm-up stays easier than the chosen target
    assert session.trainer.threshold.target_rate == pytest.approx(0.65)


def test_summary_tracks_states_and_is_saved(tmp_path):
    log = tmp_path / "run.csv"
    summary = Session(demo_device(), program=tiny(get_program("calm-anxiety")), log_path=log).run()
    assert set(summary.states) == {"concentration", "active", "calm_focus", "reflective", "drowsy", "deep_rest"}
    assert sum(summary.states.values()) == pytest.approx(summary.updates * 0.25 * (1 - summary.artifact_fraction), abs=0.5)
    saved = json.loads(log.with_suffix(".json").read_text())
    assert saved["program_id"] == "calm-anxiety" and saved["started_at"]
    assert saved["states"] == summary.states
    assert "Brain states:" in summary.format()


def test_history_and_target_api(tmp_path):
    from nova.live import DashboardServer, SessionController, load_history

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.json").write_text(json.dumps({"program": "A", "started_at": "2026-01-01T10:00:00"}))
    (tmp_path / "logs" / "b.json").write_text(json.dumps({"program": "B", "started_at": "2026-01-02T10:00:00"}))
    (tmp_path / "logs" / "bad.json").write_text("{")
    assert [h["program"] for h in load_history(tmp_path / "logs")] == ["B", "A"]

    ctl = SessionController(log_dir=tmp_path / "logs", programs_dir=tmp_path / "programs")
    assert ctl.handle_get("/api/history")[1]["sessions"][0]["program"] == "B"
    code, body = ctl.handle_post("/api/session/target", {"reward_rate": 0.5})
    assert code == 400 and "no session" in body["error"]

    server = DashboardServer(port=0)
    ctl.attach(server)
    server.start()
    try:
        ctl.start({"program": "calm-anxiety", "device": {"id": "demo"}})
        for _ in range(100):
            if ctl.session is not None:
                break
            time.sleep(0.05)
        code, body = ctl.handle_post("/api/session/target", {"reward_rate": 0.5})
        assert code == 200 and ctl.session._rate_override == 0.5
        time.sleep(2.6)  # past the first 2 s analysis window, so the session has data to save
        ctl.session.stop()
        for _ in range(100):
            if ctl.status()["state"] == "finished":
                break
            time.sleep(0.05)
        assert len(load_history(tmp_path / "logs")) == 3  # the new session was saved too
    finally:
        server.stop()
