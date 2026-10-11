"""The console fits programs to a session length in JavaScript for demo sessions; the
engine does it in Python. Both must produce the same phases."""
import json
import re
import shutil
import subprocess
from importlib.resources import files

import pytest

from nova.programs import BUILTIN, SESSION_LENGTHS


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_js_fit_matches_python(tmp_path):
    html = files("nova").joinpath("dashboard/index.html").read_text()
    fn = re.search(r"function fitPhases\(p, minutes\) \{.*?\n\}\n", html, re.S).group(0)
    programs = [p.to_dict() for p in BUILTIN]
    script = tmp_path / "fit.js"
    script.write_text(fn + f"\nconst P = {json.dumps(programs)};\n"
                      f"const out = P.map(p => {json.dumps(list(SESSION_LENGTHS))}.map(m => fitPhases(p, m).map(x => [x.name, x.kind, x.minutes])));\n"
                      "console.log(JSON.stringify(out));")
    js = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout)
    py = [[[[ph.name, ph.kind, ph.minutes] for ph in p.fitted(m).phases] for m in SESSION_LENGTHS] for p in BUILTIN]
    assert js == py
