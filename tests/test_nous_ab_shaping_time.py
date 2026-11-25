"""A shaper that cannot be served still BURNS TIME, and nothing said how much.

MEASURED 2026-11-28 on the live fleet with both cards held outside ollama's view
(``/api/ps`` answered ``{"models": []}`` on the 5090 AND the 3090, while every
model load died ``cudaMemGetInfo failed (out of memory), returning 0/0``):

    shape.shape("Summarise this task in one line")
      attempt 1: skip in 5.6 s
      attempt 2: skip in 3.5 s

So one doomed shaping dispatch costs 3.5-5.6 s, and it happens on EVERY shaped
run of an A/B. ``nous_ab.py`` recorded ``wall_s`` for the frontier call only, so
the harness could not distinguish

    "shaping is expensive and bought nothing"   (the measured situation)
    "shaping was never attempted"               (a shaper flattened out)

Both print ``shaped arm NEVER SHAPED: 0/N``. The token goal this stack exists to
serve needs the TIME figure beside it: a lever that costs minutes of wall clock
and saves zero tokens is not a saving, and it was unmeasured.

DESIGN NOTE ON THE GUARDS. Each drives ``main()`` and asserts on what it PRINTED,
because a guard reading the new ``shaping_wall_s`` field would fail on the
pre-fix tree by reading a missing key -- which bounds no behaviour. The controls
must pass on BOTH trees.

    python -m pytest tests/test_nous_ab_shaping_time.py -q
"""
import io
import json
import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "comms-lane"
sys.path.insert(0, str(LANE))

import nous_ab  # noqa: E402
import shape  # noqa: E402


class _Resp(io.BytesIO):
    def __init__(self, payload: dict):
        super().__init__(json.dumps(payload).encode())
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


BODY = {
    "choices": [{"finish_reason": "stop",
                 "message": {"content": "a substantive answer " * 20}}],
    "usage": {"completion_tokens": 40},
}


def _run(monkeypatch, tmp_path, capfd, shaper):
    def fake_urlopen(_req, timeout=None):
        return _Resp(BODY)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shape, "shape", shaper)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out


def _slow_unavailable(_task, *_a, **_kw):
    """The live box's behaviour: a load attempt that fails after ~real latency."""
    time.sleep(0.15)
    raise shape.ShaperUnavailable(
        "model 'defiant9b-abliterated:256k' is listed on the shaper host but "
        "could not be loaded (HTTP 500); the backend reported out of memory")


def _rewrites(task, *_a, **_kw):
    time.sleep(0.02)
    return "REWRITTEN: " + task + " " * 200


# ------------------------------------------------------------------ guards ---


def test_a_dead_shaper_reports_the_time_it_burned(monkeypatch, tmp_path, capfd):
    """THE DEFECT: the spend was invisible, so 0/N looked merely uninformative."""
    printed = _run(monkeypatch, tmp_path, capfd, _slow_unavailable)
    assert "time spent shaping and buying nothing" in printed, (
        "every shaped run burned a load attempt and the summary reported no "
        f"time at all:\n{printed}")


def test_the_reported_time_is_not_a_placeholder(monkeypatch, tmp_path, capfd):
    """The figure must be the measured wall time, not a hardcoded 0.0 or 1.0.

    Two tasks x one shaped run each = 2 sleeps of 0.15 s, so anything below
    0.3 s is not a measurement of what happened.
    """
    printed = _run(monkeypatch, tmp_path, capfd, _slow_unavailable)
    line = next((ln for ln in printed.splitlines()
                 if "time spent shaping and buying nothing" in ln), "")
    assert line, printed
    figure = float(line.split("nothing:")[1].split("s")[0].strip())
    assert figure >= 0.3, (
        f"reported {figure}s for two 0.15s attempts -- not a measured figure: "
        f"{line!r}")


# ----------------------------------------------------------------- controls ---


def test_ctrl_a_slow_but_successful_shaper_reports_no_waste(
        monkeypatch, tmp_path, capfd):
    """CONTROL: shaping that WORKS must not be called wasted. Passes both trees."""
    printed = _run(monkeypatch, tmp_path, capfd, _rewrites)
    assert "time spent shaping and buying nothing" not in printed, (
        f"a working shaper was reported as buying nothing:\n{printed}")
    assert "rewrote 2/2" in printed, printed


def test_ctrl_the_never_shaped_verdict_still_prints(
        monkeypatch, tmp_path, capfd):
    """CONTROL: the existing 0/N line must survive beside the new one."""
    printed = _run(monkeypatch, tmp_path, capfd, _slow_unavailable)
    assert "NEVER SHAPED" in printed, printed
    assert "0/2" in printed, printed
