"""One report said the sweep used a local GPU and did not, in the same breath.

WHY THIS EXISTS
---------------
The shaped arm dispatches to a model on one of OUR OWN CARDS before the frontier
call. That dispatch is work on a local GPU, and it is paid for.

`nous_ab.py` reported the two halves of that in two places that could not see
each other:

    shaper tokens: 474 prompt tokens in + 250 completion tokens out over 2
                   shaping dispatch(es) -- paid on the 3090, invisible to the
                   frontier count above
    endpoint: http://127.0.0.1:8099/v1/chat/completions
    all 4 run(s) went to a HOSTED lane; 0 local GPU(s) used

MEASURED 2026-12-03 by driving `main()` with the shaper stubbed to answer and
the frontier stubbed to ANSWER. Both sentences printed about the same sweep. The
lane sentence was computed from `results["endpoint"]` alone -- the FRONTIER
endpoint -- so a fully-shaped sweep to a billed proxy reported "0 local GPU(s)
used" while the shaper line said a card had been paid. That is the DISTRIBUTE
goal's number reading false in the one place a person reads it.

Two further defects on the same lines:

* "paid on the 3090" was a HARDCODED string. The shaper's endpoint is
  `shape.OLLAMA`, overridable by `SHAPE_OLLAMA`; repointing the shaper would
  leave the report still saying 3090.
* "invisible to the frontier count above" is true of the frontier arithmetic and
  false as a statement about the report, which contained the contradiction.

WHAT THE FIX DOES
-----------------
The results file gains a `shaper` block -- `endpoint`, `model`, `lane`,
`dispatches` -- written beside `lane_arrivals`, so the console and the evidence
cannot disagree. The shaper token line names the lane read from the shaper's own
endpoint instead of the string "3090". The lane sentence counts the shaper's
dispatches: a fully-shaped hosted sweep now says it DID use a local GPU and gives
both counts, and a sweep with no shaping says so rather than implying the frontier
call was the whole story.

GUARDS vs CONTROLS
------------------
GUARD (fails on the module pinned from ``git show HEAD:``):
    * TestAShapedHostedSweepDoesNotClaimNoLocalGpu
    * TestTheShaperLaneIsRecordedInTheRunFile
    * TestTheShaperLineNamesItsOwnEndpoint
    * TestAnUnshapedHostedSweepSaysNoCardWasUsed

CONTROL (passes on BOTH trees):
    * TestAHostedFrontierSweepStillSaysHosted
    * TestALocalFrontierSweepStillSaysLocal

Guards assert the PRINTED SENTENCE and the JSON KEYS, never a new helper's
name, so none fails pre-fix by ``AttributeError`` -- a missing symbol bounds no
behaviour.

    LLM_TOOLKIT_COMMS_DIR=C:/testingtemp/llmstack-1203/prefix \\
        python -m pytest tests/test_nous_ab_shaper_lane_is_counted.py -q
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMS = Path(os.environ.get("LLM_TOOLKIT_COMMS_DIR") or (REPO / "comms-lane"))
if str(COMMS) not in sys.path:
    sys.path.insert(0, str(COMMS))

import nous_ab  # noqa: E402
import shape  # noqa: E402

HOSTED = "http://127.0.0.1:8099/v1/chat/completions"
LOCAL = "http://192.168.1.232:8081/v1/chat/completions"

#: What the pre-fix tree printed for a sweep whose shaper had just used a card.
PRE_FIX_NO_LOCAL = "0 local GPU(s) used"


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


def _run(monkeypatch, tmp_path, capfd, endpoint, shaper_answers):
    """Drive main() and return (printed, saved_json).

    `shaper_answers` decides whether the shaper returns a rewrite or raises. When
    it answers, the rewrite carries a real usage dict so the shaper token line
    has figures to print -- that line is the other half of the contradiction.
    """

    def fake_urlopen(_req, timeout=None):
        return _Resp(BODY)

    def fake_shape_with_usage(task, *_a, **_kw):
        if not shaper_answers:
            raise RuntimeError("shaper unavailable")
        return (task + " [rewritten on a card]",
                {"prompt_tokens": 237, "completion_tokens": 125,
                 "expansion_x": 3.13, "basis": "server-reported"})

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shape, "shape_with_usage", fake_shape_with_usage)
    monkeypatch.setattr(nous_ab.shape, "shape_with_usage", fake_shape_with_usage)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    monkeypatch.setattr(nous_ab, "ENDPOINT", endpoint)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out, json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ guards ---


class TestAShapedHostedSweepDoesNotClaimNoLocalGpu:
    """GUARD: the sentence must not deny a card the shaper just used."""

    def test_a_shaped_hosted_sweep_does_not_claim_no_local_gpu(
            self, monkeypatch, tmp_path, capfd):
        printed, saved = _run(monkeypatch, tmp_path, capfd, HOSTED,
                              shaper_answers=True)

        assert saved["shaper"]["dispatches"] > 0, (
            "fixture did not shape anything, so this arm measures nothing:\n"
            f"{json.dumps(saved.get('shaper'))}"
        )
        assert PRE_FIX_NO_LOCAL not in printed, (
            "the lane sentence says no local GPU was used in the same report "
            "whose shaping dispatch(es) went to a local card; the two lines "
            f"contradict each other:\n{printed}"
        )

class TestTheShaperLaneIsRecordedInTheRunFile:
    """GUARD: the console and the evidence must be able to be checked against
    each other -- the same reason `lane_arrivals` is written before the file."""

    def test_the_shaper_lane_is_recorded_in_the_run_file(
            self, monkeypatch, tmp_path, capfd):
        _printed, saved = _run(monkeypatch, tmp_path, capfd, HOSTED,
                               shaper_answers=True)

        shaper = saved.get("shaper")
        assert shaper, (
            "the run file records no shaper block, so a reader of the evidence "
            "cannot see which lane the shaping half went to -- only the console "
            "said it: keys=%r" % (sorted(saved),)
        )
        assert shaper["endpoint"], "the shaper block names no endpoint"
        assert shaper["lane"] in ("local", "hosted", "unknown"), shaper
        assert isinstance(shaper["dispatches"], int), shaper

    def test_the_recorded_lane_matches_the_shapers_own_endpoint(
            self, monkeypatch, tmp_path, capfd):
        _printed, saved = _run(monkeypatch, tmp_path, capfd, HOSTED,
                               shaper_answers=True)

        assert saved["shaper"]["endpoint"] == getattr(shape, "OLLAMA", None)
        assert saved["shaper"]["lane"] == nous_ab._lane_kind(getattr(shape, "OLLAMA", None))


class TestTheShaperLineNamesItsOwnEndpoint:
    """GUARD: "paid on the 3090" was hardcoded and would lie after a repoint."""

    def test_the_shaper_line_names_its_own_endpoint(
            self, monkeypatch, tmp_path, capfd):
        printed, _saved = _run(monkeypatch, tmp_path, capfd, HOSTED,
                               shaper_answers=True)

        assert "shaper tokens:" in printed, (
            f"no shaper token line at all, so this arm cannot bound one:\n{printed}"
        )
        line = [ln for ln in printed.splitlines() if "shaper tokens:" in ln][0]
        # The fix names the endpoint it read. The pre-fix line named a model of
        # card by hand ("the 3090") and said the spend was invisible to the
        # report, which the report itself disproved.
        assert getattr(shape, "OLLAMA", None) in line or "LOCAL lane" in line, (
            "the shaper token line neither names the shaper's own endpoint nor "
            f"calls it the local lane: {line!r}"
        )
        assert "invisible to the frontier count above" not in line, (
            "the line still calls the shaper's spend invisible to the report, "
            "which printed the contradiction: " + line
        )


class TestAnUnshapedHostedSweepSaysNoCardWasUsed:
    """GUARD: when nothing was shaped, the report must say that rather than let
    "0 local GPU(s) used" imply the frontier call was the whole sweep."""

    def test_an_unshaped_hosted_sweep_says_no_card_was_used(
            self, monkeypatch, tmp_path, capfd):
        printed, saved = _run(monkeypatch, tmp_path, capfd, HOSTED,
                              shaper_answers=False)

        assert saved["shaper"]["dispatches"] == 0, saved["shaper"]
        # Pre-fix this printed the bare sentence with no statement about
        # shaping, which reads as "the sweep was one lane" rather than "the
        # other lane was never entered".
        assert "no run was shaped" in printed or "never shaped" in printed or (
            "shaping dispatched to no card" in printed), (
            f"an unshaped sweep said nothing about shaping:\n{printed}"
        )


# ---------------------------------------------------------------- controls ---


class TestAHostedFrontierSweepStillSaysHosted:
    """CONTROL: green on both trees -- the hosted verdict survives the change."""

    def test_a_hosted_frontier_sweep_still_says_hosted(
            self, monkeypatch, tmp_path, capfd):
        printed, _saved = _run(monkeypatch, tmp_path, capfd, HOSTED,
                               shaper_answers=False)

        assert "HOSTED" in printed, printed
        assert not nous_ab._is_hosted(LOCAL), (
            "the control's own premise is wrong: LOCAL is not a local endpoint"
        )


class TestALocalFrontierSweepStillSaysLocal:
    """CONTROL: green on both trees -- a card sweep still reads as a card."""

    def test_a_local_frontier_sweep_still_says_local(
            self, monkeypatch, tmp_path, capfd):
        printed, _saved = _run(monkeypatch, tmp_path, capfd, LOCAL,
                               shaper_answers=False)

        assert "LOCAL lane" in printed, printed
        assert PRE_FIX_NO_LOCAL not in printed, (
            "a local-card sweep was reported as having used no local GPU:\n"
            f"{printed}"
        )


# ------------------------------------------------------------------- pinned --


def test_the_negative_control_is_pointed_somewhere():
    """Guard-of-the-guards: name the module under test in the failure output."""
    assert (COMMS / "nous_ab.py").exists(), (
        "no nous_ab.py under %s -- every arm above measured nothing" % (COMMS,)
    )
    override = os.environ.get("LLM_TOOLKIT_COMMS_DIR")
    if override and not Path(override).exists():
        pytest.fail("the pinned comms dir does not exist: %s" % (override,))
