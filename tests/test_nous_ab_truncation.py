"""A budget-cut reasoning model is not a failed call.

The comms lane promises a MEASURED refusal/answer delta (ROUTING.md rule 4), and
nous_ab.py is the harness that produces it. It classified every run with an empty
`message.content` as ERROR -- a verdict about the CALL -- when the cause was the
BUDGET. finish_reason was already in the response it stored, and was never read.

MEASURED 2026-11-27 against 127.0.0.1:8099, model z-ai/glm-5.3-flashx, this
harness's own task text, one call per budget:

    budget  finish  content  reasoning  ctoks  %budget-on-reasoning
    1200    length        0      5,478   1200    100%   <- old AB_MAX_TOKENS
    2000    stop      4,357      1,227   1255     18%
    3000    stop      3,051      2,091   1088     37%
    4000    stop      2,642      3,130   1341     50%
    6000    length        0     25,416   6000    100%

At 1200 the model spends the ENTIRE cap thinking and writes nothing. The saved
comms-lane/ab-nous-results.json therefore reads 4/4 ERROR on four runs that each
burned 1,200 completion tokens. A reader would conclude the shaping lever could
not be measured, when the finding is that the budget was too small and 4,800
tokens were spent to produce nothing.

Three states, three different facts:
    finish_reason=length, no content  -> TRUNCATED  (budget ran out first)
    finish_reason=stop,   no content  -> EMPTY      (had room, answered nothing)
    the call raised                   -> ERROR      (the lane failed)

DESIGN NOTE ON THE GUARDS. Each drives `call()` (or the rendered summary) and
asserts on its OUTPUT -- the returned status, or the printed line. None calls a
helper this fix introduces: such a guard would fail on the pre-fix tree by
AttributeError, which is a missing symbol and bounds no behaviour.

A CONTROL THAT WAS WRONG, KEPT AS A NOTE. The first draft asserted
`wasted_completion_tokens == 0` for a healthy answer under the controls heading.
Measured on the HEAD-only tree, it FAILED there -- the key does not exist, so it
was measuring absence, not conduct. A control must pass on BOTH trees; anything
that names a field the fix adds is a guard wearing the wrong name. Removed, and
the token-waste claim now rides on the RENDERED summary line instead, which is a
consequence.

    python -m pytest tests/test_nous_ab_truncation.py -q
"""
import io
import json
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "comms-lane"
sys.path.insert(0, str(LANE))

import nous_ab  # noqa: E402


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


def _body(*, finish, content="", reasoning="", completion_tokens=1200,
          reasoning_tokens=None):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    return {
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": (completion_tokens if reasoning_tokens
                                     is None else reasoning_tokens)},
        },
    }


def _run(monkeypatch, tmp_path, payload=None, *, raises=None):
    def fake_urlopen(_req, timeout=None):
        if raises is not None:
            raise raises
        return _Resp(payload)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    return nous_ab.call("task text", tmp_path / "out.json")


# ------------------------------------------------------------------ guards ---
# Each reads ERROR on the pre-fix tree, which is the defect: a verdict that
# blames the call for what the budget did.

def test_a_run_cut_off_by_the_budget_is_not_reported_as_a_failed_call(monkeypatch, tmp_path):
    """THE EXACT SHAPE the live lane returns at the old AB_MAX_TOKENS=1200."""
    row = _run(monkeypatch, tmp_path,
               _body(finish="length", content="", reasoning="x" * 5478,
                     completion_tokens=1200))
    assert row["status"] != "ERROR", (
        f"finish_reason=length with no content was reported as {row['status']!r}; "
        "the model was cut off, not broken")
    assert row["status"] == "TRUNCATED", row["status"]


def test_an_empty_answer_with_room_to_answer_is_not_a_truncation(monkeypatch, tmp_path):
    """"It ran out of budget" and "it had room and said nothing" are different.

    Only the first is fixed by raising AB_MAX_TOKENS, so collapsing them hides
    which lever applies.
    """
    row = _run(monkeypatch, tmp_path,
               _body(finish="stop", content="", completion_tokens=40))
    assert row["status"] != "ERROR", (
        f"an empty answer with finish_reason=stop was reported as {row['status']!r}")
    assert row["status"] == "EMPTY", row["status"]
    truncated = _run(monkeypatch, tmp_path,
                     _body(finish="length", content="", completion_tokens=1200))
    assert row["status"] != truncated["status"], (
        "a truncated run and a genuinely empty one report the same status")


def test_the_summary_states_the_tokens_a_truncation_wasted(monkeypatch, tmp_path, capsys):
    """The rendered report must put the wasted tokens on the operator's screen.

    Without this line a fully-truncated sweep and a lane outage print the same
    shape -- every row non-answering -- and the reader is pushed to the wrong
    lever. On the pre-fix tree the phrase is absent entirely, so this guard
    measures the rendered output, not a new symbol.
    """
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    monkeypatch.setattr(nous_ab, "TASKS", [{"name": "t", "label": "t", "raw": "go"}])
    monkeypatch.setattr(nous_ab.shape, "shape", lambda _t: "")  # no 3090 in tests
    monkeypatch.setattr(
        nous_ab.urllib.request, "urlopen",
        lambda _req, timeout=None: _Resp(
            _body(finish="length", content="", completion_tokens=1200)))
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(tmp_path / "r.json")])

    assert nous_ab.main() == 0
    out = capsys.readouterr().out
    # one task x raw+shaped = 2 runs x 1200 tokens, all producing no text
    assert "completion tokens spent with ZERO usable output: 2400" in out, out
    assert "raise AB_MAX_TOKENS" in out, out


# ---------------------------------------------------------------- controls ---
# These pass on both trees. The fix must not turn a real answer into a
# non-answer, nor a real call failure into a soft one.

def test_a_real_answer_still_reads_as_an_answer(monkeypatch, tmp_path):
    row = _run(monkeypatch, tmp_path,
               _body(finish="stop", content="Use a flat huge-page buffer as a "
                     "lookup table. " * 20, completion_tokens=120))
    assert row["status"] == "ANSWER", row["status"]


def test_a_refusal_is_still_a_refusal(monkeypatch, tmp_path):
    row = _run(monkeypatch, tmp_path,
               _body(finish="stop", content="I cannot help with that.",
                     completion_tokens=12))
    assert row["status"] == "REFUSE", row["status"]


def test_a_call_that_raises_is_still_an_error(monkeypatch, tmp_path):
    """The one thing ERROR must keep meaning: the lane itself failed."""
    row = _run(monkeypatch, tmp_path,
               raises=urllib.error.URLError("connection refused"))
    assert row["status"] == "ERROR", row["status"]
    assert "error" in row, row
