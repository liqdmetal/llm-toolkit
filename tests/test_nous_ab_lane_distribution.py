"""The A/B summary named the model and never the lane that took the work.

`nous_ab.py` writes `endpoint` into every result file it saves. Nothing in the
repo read that key: it was written at one site in `main()` and read at NONE, so
the summary printed `=== SUMMARY (model) ===` and stopped there. Which lane
answered was in the JSON and invisible on the console.

That is the DISTRIBUTE goal's defect and it is not theoretical. MEASURED
2026-11-28 over the four saved A/B files in `comms-lane/`:

    ab-nous-qwen.json     endpoint=...:8099...   runs=4
    ab-nous-results.json  endpoint=...:8099...   runs=4
    ab-nous-solar.json    endpoint=...:8099...   runs=4
    ab-results.json       endpoint=None          runs=1

    13 runs total: 12 to the hosted nous lane, 0 to any local GPU, and 1 whose
    file records no endpoint at all.

Twelve to hosted and zero to local while q38 and the 3090 both sat idle is the
one-lane-takes-everything shape this stack exists to fix, and a reader of the
summary could not tell it from a sweep that used the cards.

DESIGN NOTE ON THE GUARDS. Each drives `main()` and asserts on what it PRINTED.
None reads `_is_hosted` -- a guard naming a helper the fix introduces fails on
the pre-fix tree by AttributeError, which is a missing symbol and bounds no
behaviour. The controls must pass on BOTH trees.

    python -m pytest tests/test_nous_ab_lane_distribution.py -q
"""
import io
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "comms-lane"
sys.path.insert(0, str(LANE))

import nous_ab  # noqa: E402
import shape  # noqa: E402


class _Resp(io.BytesIO):
    """What urlopen() returns: a context manager over the body bytes."""

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

HOSTED = "http://127.0.0.1:8099/v1/chat/completions"
LOCAL = "http://192.168.1.232:8081/v1/chat/completions"


def _rewrites(task, *_a, **_kw):
    return "REWRITTEN: " + task + " " * 200


def _run(monkeypatch, tmp_path, capfd, endpoint):
    """Drive main() against a fake successful call on `endpoint`."""

    def fake_urlopen(_req, timeout=None):
        return _Resp(BODY)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shape, "shape", _rewrites)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    # ENDPOINT is read inside main() from the module global, so this is the
    # one place the lane is decided.
    monkeypatch.setattr(nous_ab, "ENDPOINT", endpoint)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out


# ------------------------------------------------------------------ guards ---
# Each reads on the pre-fix tree as "no such line", which is the defect: the
# endpoint was saved to JSON and never printed, so the lane was not stated.

def test_the_summary_states_the_endpoint(monkeypatch, tmp_path, capfd):
    """The lane must be on the console, not only in the saved JSON."""
    printed = _run(monkeypatch, tmp_path, capfd, HOSTED)
    assert HOSTED in printed, (
        "the summary never named the endpoint it dispatched to; a reader "
        f"cannot tell hosted from local:\n{printed}")


def test_a_fully_hosted_sweep_says_no_local_gpu_was_used(
        monkeypatch, tmp_path, capfd):
    """THE MEASURED SHAPE: 12 of 12 hosted, 0 local, said out loud."""
    printed = _run(monkeypatch, tmp_path, capfd, HOSTED)
    lowered = printed.lower()
    assert "hosted" in lowered, printed
    assert "0 local" in lowered, (
        "every run went to a hosted lane and the summary did not say so; that "
        f"is the deflection this line exists to expose:\n{printed}")


def test_a_local_sweep_is_not_reported_as_hosted(monkeypatch, tmp_path, capfd):
    """The anti-vacuity arm on the guard side: the line must be able to differ.

    A fix that printed the same sentence for every endpoint would satisfy both
    guards above. This arm requires the hosted verdict to be ABSENT when the
    lane was a local card.
    """
    printed = _run(monkeypatch, tmp_path, capfd, LOCAL)
    assert LOCAL in printed, printed
    assert "0 local" not in printed.lower(), (
        "a local-card sweep was reported as having used no local GPU:\n"
        f"{printed}")


# ---------------------------------------------------------------- controls ---
# These pass on the pre-fix tree as well. A control failing pre-fix would be a
# guard mislabelled, and a control passing only post-fix would bound nothing.

def test_the_summary_still_names_the_model(monkeypatch, tmp_path, capfd):
    """CONTROL -- the fix must not displace what the summary already said."""
    printed = _run(monkeypatch, tmp_path, capfd, HOSTED)
    assert nous_ab.MODEL in printed, printed
    assert "SUMMARY" in printed, printed


def test_the_run_file_is_still_written_and_carries_the_endpoint(
        monkeypatch, tmp_path, capfd):
    """CONTROL -- the recorded evidence must survive the console change."""
    _run(monkeypatch, tmp_path, capfd, LOCAL)
    doc = json.loads((tmp_path / "ab.json").read_text(encoding="utf-8"))
    assert doc["endpoint"] == LOCAL
    assert doc["model"] == nous_ab.MODEL
    assert doc["runs"], "no runs recorded"


def test_a_normal_run_still_exits_zero(monkeypatch, tmp_path, capfd):
    """CONTROL -- _run already asserts == 0; this pins it as the called shape."""
    printed = _run(monkeypatch, tmp_path, capfd, HOSTED)
    assert "wrote" in printed, printed
