"""An A/B whose shaped arm never shaped is not a comparison.

`nous_ab.py` exists to produce a MEASURED refusal/answer delta between a raw
prompt and a shaped one (ROUTING.md rule 4). The shaped arm calls
`shape.shape()`, and ROUTING.md rule 1 makes shaping a soft dependency: any
`ShaperUnavailable` falls back to the raw text.

That fallback is correct. What was missing is that nothing said it happened. The
fallback is recorded in each run's free-text `note`, and the summary never read
it, so the two arms could send byte-identical prompts while the summary printed
`raw=OK shaped=OK` for every task.

MEASURED 2026-11-28 on the live box, with Inkling holding the 3090 and
`defiant9b-abliterated:256k` listed but unloadable:

    shape.shape("Summarise the routing rules.")
      -> ShaperUnavailable: model 'defiant9b-abliterated:256k' is listed on the
         shaper host but could not be loaded

and in the harness, 4 of 4 shaped runs fell back to raw with
`prompt_chars` identical across the two arms. A reader diffing those arms
measures 0.0 and concludes shaping does not help, when the truth is that shaping
never ran. That is the token-cost claim this stack exists to keep honest.

DESIGN NOTE ON THE GUARDS. Each drives `main()` and asserts on what it PRINTED.
None reads a helper this fix introduces, and none reads `shaped_applied` -- a
guard naming a field the fix adds fails on the pre-fix tree by KeyError, which is
a missing symbol and bounds no behaviour. The control must pass on BOTH trees.

    python -m pytest tests/test_nous_ab_shaping_ran.py -q
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


def _run(monkeypatch, tmp_path, capfd, shaper):
    """Drive main() with a fake shaper and a fake successful HTTP call."""

    def fake_urlopen(_req, timeout=None):
        return _Resp(BODY)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shape, "shape", shaper)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out


def _unavailable(_task, *_a, **_kw):
    raise shape.ShaperUnavailable(
        "model 'defiant9b-abliterated:256k' is listed on the shaper host but "
        "could not be loaded (HTTP 500)")


def _rewrites(task, *_a, **_kw):
    return "REWRITTEN: " + task + " " * 200


# ------------------------------------------------------------------ guards ---
# Each reads on the pre-fix tree as "no such line", which is the defect: the
# summary reported the two arms as a comparison when they were the same prompt.

def test_a_shaped_arm_that_never_shaped_is_reported_as_no_comparison(
        monkeypatch, tmp_path, capfd):
    """THE EXACT SHAPE the live box produces: shaper listed, card unavailable."""
    printed = _run(monkeypatch, tmp_path, capfd, _unavailable)
    assert "NEVER SHAPED" in printed, (
        "every shaped run fell back to the raw text and the summary said "
        f"nothing; a reader measures 0.0 as a shaping effect:\n{printed}")


def test_the_warning_names_the_cause_and_the_count(monkeypatch, tmp_path, capfd):
    printed = _run(monkeypatch, tmp_path, capfd, _unavailable)
    line = next((ln for ln in printed.splitlines() if "NEVER SHAPED" in ln), "")
    assert "0/2" in line, line          # two tasks, no rewrites
    assert "could not be loaded" in printed or "shaper" in line, line


def test_a_shape_that_ran_says_so_and_does_not_warn(monkeypatch, tmp_path, capfd):
    """The line is conditional: a working shaper must not print a warning."""
    printed = _run(monkeypatch, tmp_path, capfd, _rewrites)
    assert "NEVER SHAPED" not in printed, (
        f"a working shaper was reported as never shaping:\n{printed}")
    assert "rewrote 2/2" in printed, printed


def test_a_mixed_run_is_called_mixed_not_pure(monkeypatch, tmp_path, capfd):
    """One task shaped, one fell back: neither "never" nor a clean comparison."""
    calls = {"n": 0}

    def sometimes(task, *_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _rewrites(task)
        return _unavailable(task)

    printed = _run(monkeypatch, tmp_path, capfd, sometimes)
    assert "NEVER SHAPED" not in printed, printed
    assert "rewrote 1/2" in printed, printed
    assert "mixed" in printed, printed


# ----------------------------------------------------------------- controls ---
# These must pass on the pre-fix tree too, so they assert only that the harness
# still does its ordinary job.

def test_the_arms_still_run_and_the_receipt_is_written(monkeypatch, tmp_path, capfd):
    printed = _run(monkeypatch, tmp_path, capfd, _rewrites)
    assert "ANSWER" in printed, printed
    receipt = json.loads((tmp_path / "ab.json").read_text(encoding="utf-8"))
    assert len(receipt["runs"]) == 4, receipt["runs"]
    assert {r["condition"] for r in receipt["runs"]} == {"raw", "shaped"}


def test_a_failing_lane_is_reported_as_failed_calls(
        monkeypatch, tmp_path, capfd):
    """A dead lane must reach its own summary line.

    THE DEFECT THIS FOUND: call() returns no 'head' key when the request
    raised, and main() printed r['head'] unconditionally, so the FIRST failed
    call killed main() with a KeyError -- before 'failed calls: N/N' could
    print. The one line that explains a lane outage was unreachable exactly
    when it was needed, and the harness exited on a traceback instead of a
    receipt.
    """
    def boom(_req, timeout=None):
        raise RuntimeError("lane down")

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", boom)
    monkeypatch.setattr(shape, "shape", _rewrites)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["nous_ab.py", "--out", str(tmp_path / "ab.json")])
    assert nous_ab.main() == 0
    printed = capfd.readouterr().out
    assert "failed calls: 4/4" in printed, printed
    # The receipt is written even when every call failed.
    receipt = json.loads((tmp_path / "ab.json").read_text(encoding="utf-8"))
    assert len(receipt["runs"]) == 4
