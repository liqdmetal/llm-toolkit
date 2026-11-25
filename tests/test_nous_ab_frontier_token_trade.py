"""The frontier INPUT side of the token trade was measured and thrown away.

`call()` wrote the whole API response to disk -- which carries `prompt_tokens`
and `completion_tokens_details.reasoning_tokens` -- and then copied exactly one
number out of `usage`: `completion_tokens`. So the harness that exists to answer
"what does shaping cost" could not state the cost of the prompt shaping *sends*.

MEASURED 2026-11-28 over the four saved responses in the scratch out-dir,
z-ai/glm-5.3-flashx, this file's own two tasks:

    task     prompt        completion       reasoning
    framing  60 -> 108     2000 -> 640      1998 -> 395
    giant    58 -> 102     2000 -> 2000     1992 -> 2000

Shaping ADDS 76-80% to the frontier prompt on both tasks, and on "giant" it
saved nothing at all: the two arms spent identical completion tokens. The
summary printed `wasted_completion_tokens` only, which answers a different
question (was the BUDGET big enough), so the "decrease token usage" goal had no
figure it could move.

DESIGN NOTE ON THE GUARDS. They drive `main()` and assert on what it PRINTED,
and they read the saved run record by VALUE (`.get(...) != 102`), so a guard
fails on the pre-fix tree because the number is absent, not because a key raised
KeyError. Controls pass on BOTH trees.

    python -m pytest tests/test_nous_ab_frontier_token_trade.py -q
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


# The two arms get DIFFERENT counts, decided by the prompt the request actually
# carries -- not by call order -- so the arms cannot be mixed up by a change to
# the loop.
RAW_PROMPT, SHAPED_PROMPT = 58, 102
COMPLETION = 2000


def _body_for(shaped: bool) -> dict:
    return {
        "choices": [{"finish_reason": "stop",
                     "message": {"content": "a substantive answer " * 20,
                                 "reasoning": "thinking " * 30}}],
        "usage": {
            "prompt_tokens": SHAPED_PROMPT if shaped else RAW_PROMPT,
            "completion_tokens": COMPLETION,
            "completion_tokens_details": {"reasoning_tokens": 1992},
        },
    }


def _run(monkeypatch, tmp_path, capfd, *, shaper, drop_prompt_tokens=False):
    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode())
        content = payload["messages"][0]["content"]
        shaped = content.startswith("REWRITTEN:")
        body = _body_for(shaped)
        if drop_prompt_tokens:
            del body["usage"]["prompt_tokens"]
        return _Resp(body)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shape, "shape", shaper)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    printed = capfd.readouterr().out
    return printed, json.loads(out.read_text(encoding="utf-8"))


def _rewrites(task, *_a, **_kw):
    return "REWRITTEN: " + task


def _falls_back(_task, *_a, **_kw):
    raise RuntimeError("a card held by another model")


# -- guards: these FAIL on the unmodified harness ----------------------------


def test_the_summary_states_prompt_tokens_per_arm(monkeypatch, tmp_path, capfd):
    """PIN -- measured: the summary never named the frontier input side."""
    printed, _doc = _run(monkeypatch, tmp_path, capfd, shaper=_rewrites)
    assert "frontier prompt tokens" in printed, printed
    # Both arms named, with the increase stated: the point is that shaping costs
    # prompt tokens, so a line reporting only the shaped figure would hide the
    # comparison the goal needs.
    line = next(l for l in printed.splitlines()
                if "frontier prompt tokens" in l)
    assert str(RAW_PROMPT * 2) in line, line
    assert str(SHAPED_PROMPT * 2) in line, line
    assert f"+{(SHAPED_PROMPT - RAW_PROMPT) * 2}" in line, line


def test_the_summary_states_the_net_token_change(monkeypatch, tmp_path, capfd):
    """PIN -- one arm can gain on input and lose on output; only a net says so."""
    printed, _doc = _run(monkeypatch, tmp_path, capfd, shaper=_rewrites)
    assert "frontier tokens in total" in printed, printed
    net = (SHAPED_PROMPT + COMPLETION) * 2 - (RAW_PROMPT + COMPLETION) * 2
    assert f"{net:+d}" in printed, printed


def test_each_run_record_carries_the_prompt_token_count(monkeypatch, tmp_path,
                                                        capfd):
    """PIN -- the number was in the saved response and copied out of it away."""
    _printed, doc = _run(monkeypatch, tmp_path, capfd, shaper=_rewrites)
    for run in doc["runs"]:
        expected = SHAPED_PROMPT if run["condition"] == "shaped" else RAW_PROMPT
        assert run.get("prompt_tokens") == expected, run


def test_each_run_record_carries_the_reasoning_token_split(monkeypatch,
                                                           tmp_path, capfd):
    """PIN -- the response states reasoning tokens; only the char COUNT was kept.

    Pinned separately from prompt_tokens because it is a different field with a
    different shape (`completion_tokens_details.reasoning_tokens`), and because
    the reasoning-spend line above it is in CHARS only: with this recorded the
    split is finally available in tokens too.
    """
    _printed, doc = _run(monkeypatch, tmp_path, capfd, shaper=_rewrites)
    for run in doc["runs"]:
        assert run.get("reasoning_tokens") == 1992, run


# -- controls: these pass on BOTH trees --------------------------------------


def test_a_shaper_that_falls_back_prints_no_trade(monkeypatch, tmp_path, capfd):
    """CONTROL -- when the shaped arm sent raw text there is no delta to state.

    Both arms then carry the same prompt, so a printed "raw -> shaped" line
    would be the lane's own drift dressed as a shaping effect. Same rule the
    NEVER SHAPED line already enforces one paragraph above.
    """
    printed, _doc = _run(monkeypatch, tmp_path, capfd, shaper=_falls_back)
    assert "shaped arm NEVER SHAPED" in printed, printed
    assert "frontier prompt tokens" not in printed, printed


def test_an_absent_prompt_token_count_is_not_zero_filled(monkeypatch, tmp_path,
                                                         capfd):
    """CONTROL -- an absent count is not a measured zero.

    With `prompt_tokens` missing from the response the trade line must not print
    a fabricated 0. Passes on both trees: pre-fix prints no such line at all,
    post-fix prints none without a number.
    """
    printed, doc = _run(monkeypatch, tmp_path, capfd, shaper=_rewrites,
                        drop_prompt_tokens=True)
    assert "frontier prompt tokens" not in printed, printed
    for run in doc["runs"]:
        assert run.get("prompt_tokens") is None, run


def test_the_reasoning_spend_line_still_reports(monkeypatch, tmp_path, capfd):
    """CONTROL -- the pre-existing reasoning-spend report is not displaced."""
    printed, _doc = _run(monkeypatch, tmp_path, capfd, shaper=_rewrites)
    assert "reasoning spend" in printed, printed
