"""The shaper spends tokens, and the A/B that uses it could not say so.

`shape.py` ships `shape_with_usage()` for exactly one reason, stated in its own
docstring: `shape()` returned a string and threw the rest of the ollama response
away, and that response carries `prompt_eval_count`, `eval_count` and the
durations -- the actual token cost of shaping. So no caller could report what the
shaping step spends, and "shaping saves frontier tokens" was an unmeasured claim.

MEASURED on this tree (2026-12-02):

    grep -rn "shape_with_usage" --include=*.py .    ->  2 hits, both in shape.py
                                                        (its def, and shape()'s
                                                         own internal call)
    grep -rn "shape.shape(" --include=*.py .        ->  1 hit, nous_ab.py:236

So the one harness that runs a raw arm against a shaped arm calls `shape()`, the
function that discards the usage, and the A/B summary reports frontier tokens
only. On the recorded live figure from shape.py's own docstring -- 237 prompt
tokens in, 125 completion tokens out per shaping dispatch -- a four-task sweep
spends roughly 950 in / 500 out on shaping that nothing in the summary accounts
for. The "decrease token usage" goal cannot be steered by a number the tool
refuses to print.

This is deliberate about WHICH way the trade is misreported. Shaping does not
merely add a cost line: the measured rewrite is 3.13x the input's characters, so
shaping ADDS prompt tokens to save completion tokens. A summary that reports the
frontier side and hides the shaper side can only ever make shaping look better
than it is.

THE BAR THESE ARMS ENFORCE: when the shaped arm actually shaped, the run record
carries what the shaper spent and the summary states it. When the shaper could
not be reached, no figure is invented.

DESIGN NOTE ON THE GUARDS

Each drives ``main()`` and asserts on what it PRINTED and what it RECORDED, not
on a symbol this change introduces: a guard importing a new name would fail on
the pre-fix tree with ImportError and bound no behaviour. The defect is that the
run reported nothing, so the assertion belongs on the run's output. Each control
passes on BOTH trees.
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
    def __init__(self, payload: dict):
        super().__init__(json.dumps(payload).encode())
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _frontier_body() -> dict:
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "a substantive answer", "reasoning": ""},
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
    }


# What a real shaper reports. prompt_eval_count/eval_count are the two numbers
# shape.py's docstring measured against the live box.
SHAPER_USAGE = {
    "model": "defiant9b-abliterated:256k",
    "basis": "measured",
    "prompt_tokens": 237,
    "completion_tokens": 125,
    "task_chars": 199,
    "shaped_chars": 623,
    "expansion_x": 3.13,
    "produced_text": True,
}


def _run(monkeypatch, tmp_path, capfd, *, usage=None, unavailable=False):
    """Drive main() with a working frontier, a working shaper, and a usage dict.

    Both `shape()` and `shape_with_usage()` are patched, so the SAME fixture
    describes the pre-fix tree (which calls shape()) and the fixed tree (which
    calls shape_with_usage()). That is what lets a control pass on both.
    """
    def fake_urlopen(_req, timeout=None):
        return _Resp(_frontier_body())

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)

    rewritten = "REWRITTEN: "

    if unavailable:
        def _raise(*_a, **_kw):
            raise shape.ShaperUnavailable("shaper host unreachable (fixture)")
        monkeypatch.setattr(shape, "shape", _raise)
        monkeypatch.setattr(shape, "shape_with_usage", _raise)
    else:
        monkeypatch.setattr(shape, "shape",
                            lambda task, *_a, **_kw: rewritten + task)
        monkeypatch.setattr(shape, "shape_with_usage",
                            lambda task, *_a, **_kw: (rewritten + task,
                                                      dict(usage or SHAPER_USAGE)))

    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out, out


def _shaped_runs(out):
    return [r for r in json.loads(out.read_text(encoding="utf-8"))["runs"]
            if r.get("condition") == "shaped"]


# ------------------------------------------------------------------ guards ---

def test_a_shaped_arm_that_shaped_reports_what_the_shaper_spent(
        monkeypatch, tmp_path, capfd):
    """THE DEFECT: the shaper spent 237+125 tokens per dispatch and nothing said so."""
    printed, _out = _run(monkeypatch, tmp_path, capfd)
    assert "shaper" in printed.lower(), (
        "the shaped arm dispatched to a shaper that reported its own token "
        f"counts, and the summary mentioned no shaper spend at all:\n{printed}"
    )
    assert "token" in printed.lower(), printed


def test_the_shaper_figure_is_measured_not_placed(monkeypatch, tmp_path, capfd):
    """The printed total must equal the sum over the runs, not a plausible number."""
    printed, out = _run(monkeypatch, tmp_path, capfd)
    runs = _shaped_runs(out)
    assert runs, "precondition: the fixture produced shaped runs"

    recorded_in = sum(r.get("shaper_prompt_tokens") or 0 for r in runs)
    recorded_out = sum(r.get("shaper_completion_tokens") or 0 for r in runs)
    assert recorded_in > 0, (
        "no shaped run recorded what the shaper spent on its prompt, so the "
        "harness still calls the function that discards usage"
    )
    assert recorded_in == 237 * len(runs), (
        f"expected {237 * len(runs)} prompt tokens over {len(runs)} shaped "
        f"runs, records hold {recorded_in}"
    )
    line = next((ln for ln in printed.splitlines()
                 if "shaper" in ln.lower() and str(recorded_in) in ln), "")
    assert line, (
        f"the records hold {recorded_in} shaper prompt tokens and no printed "
        f"line states that figure:\n{printed}"
    )


def test_the_trade_states_which_side_shaping_adds_to(monkeypatch, tmp_path, capfd):
    """Shaping ADDS prompt tokens. A one-sided report can only flatter it."""
    printed, _out = _run(monkeypatch, tmp_path, capfd)
    low = printed.lower()
    assert "add" in low or "expansion" in low or "x the" in low, (
        "the measured rewrite is 3.13x the input's characters, so shaping adds "
        "prompt tokens to save completion tokens; the summary stated no "
        f"direction for the trade:\n{printed}"
    )


# ---------------------------------------------------------------- controls ---

def test_an_unreachable_shaper_invents_no_figure(monkeypatch, tmp_path, capfd):
    """A soft dependency that failed spent nothing and must claim nothing."""
    printed, out = _run(monkeypatch, tmp_path, capfd, unavailable=True)
    runs = _shaped_runs(out)
    assert runs, "precondition: the fixture produced shaped runs"
    claimed = [r for r in runs
               if (r.get("shaper_prompt_tokens") or 0) > 0]
    assert not claimed, (
        "the shaper could not be reached and the run still recorded a shaper "
        f"token count: {claimed[:1]}"
    )


def test_a_zero_shaper_count_is_not_a_default(monkeypatch, tmp_path, capfd):
    """An absent usage key must be absent, never zero-filled."""
    partial = {"model": "defiant9b-abliterated:256k", "basis": "measured",
               "task_chars": 199, "shaped_chars": 623, "produced_text": True}
    printed, out = _run(monkeypatch, tmp_path, capfd, usage=partial)
    runs = _shaped_runs(out)
    assert runs, "precondition: the fixture produced shaped runs"
    for r in runs:
        assert r.get("shaper_prompt_tokens") is None, (
            "the shaper reported no prompt token count and the record holds a "
            f"figure anyway: {r.get('shaper_prompt_tokens')!r}"
        )
        assert r.get("shaper_completion_tokens") is None, (
            "the shaper reported no completion token count and the record holds "
            f"a figure anyway: {r.get('shaper_completion_tokens')!r}"
        )
    assert "shaper" not in printed.lower() or "not reported" in printed.lower(), (
        "no shaper token count was reported and the summary still stated a "
        f"spend:\n{printed}"
    )


def test_the_frontier_token_trade_still_prints(monkeypatch, tmp_path, capfd):
    """The existing report must survive; this change adds a side, not replaces one."""
    printed, _out = _run(monkeypatch, tmp_path, capfd)
    assert "frontier" in printed.lower(), (
        f"the frontier token report disappeared:\n{printed}"
    )
