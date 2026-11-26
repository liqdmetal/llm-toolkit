"""The DISTRIBUTE line counted ATTEMPTS and called them arrivals.

WHY THIS EXISTS
---------------
`nous_ab.py`'s last summary line is the one place the DISTRIBUTE goal is read
from: did bulk work go to a card, or to a billed lane? It was computed as

    lane = _lane_kind(results["endpoint"])
    total = len(results["runs"])
    ...
    print(f"all {total} run(s) went to a HOSTED lane; 0 local GPU(s) used")

`total` is the number of run RECORDS, which is the number of attempts. `call()`
writes exactly one record per attempt whether or not the request reached
anything, and on a request-level failure it returns `status="ERROR"` with an
`error` string and no other fields. So the line reported a lane's NAME as
confidently for a sweep where nothing got through as for one where everything
did.

MEASURED 2026-12-03 by driving `main()` itself with `nous_ab.call` stubbed --
the real loop, the real summary, only the dispatch replaced:

    sweep                       printed
    4 runs, all ERROR           all 4 run(s) went to a HOSTED lane;
      (Connection refused)      0 local GPU(s) used      <- same line as below
    4 runs, all ANSWER          all 4 run(s) went to a HOSTED lane;
                                0 local GPU(s) used
    0 runs                      all 0 run(s) went to a HOSTED lane;
                                0 local GPU(s) used

Three collapses on one line:

  (1) A dead endpoint read as a measured placement. "0 local GPU(s) used" is a
      statement about where work went; over a sweep where no call arrived it is
      a statement about nothing, and it is the folded value of the same defect
      this repo keeps finding -- nothing, dressed as an answer.
  (2) "all 0 run(s) went to a HOSTED lane" is a placement claim about a sweep
      that dispatched nothing. `all` over an empty set is vacuous, and the
      sentence around it is not.
  (3) In a mixed sweep nothing said HOW MANY of the runs the lane sentence
      covered, so a reader could not tell 4/4 answered from 1/4.

WHAT THE FIX DOES
-----------------
Runs are split before the file is written, and the split is recorded in the
run JSON as `lane_arrivals` so the console and the evidence agree:

    ARRIVED_STATUSES = {ANSWER, REFUSE, PARTIAL, EMPTY, TRUNCATED}
    everything else  = never reached the endpoint, with a named cause

The lane sentence now counts ARRIVALS. When nothing arrived, no lane claim is
made at all; when some arrived, the count of the ones that did not is printed
first, with causes, so the lane line cannot be read as covering them.

GUARDS vs CONTROLS
------------------
GUARD (fails on the pre-fix tree):
    * TestAnAllErrorSweepMakesNoLaneClaim
    * TestAZeroRunSweepMakesNoLaneClaim
    * TestAMixedSweepSaysHowManyArrived
    * TestTheRunFileRecordsTheArrivalSplit
    * TestCausesAreNamedPerRun

CONTROL (passes on BOTH trees):
    * TestASuccessfulHostedSweepStillSaysHosted
    * TestASuccessfulLocalSweepStillSaysLocal
    * TestTheSummaryStillNamesModelEndpointAndWritesTheFile
    * TestAnArrivedRunIsNeverCountedAsMissed

The negative control points ``LLM_TOOLKIT_COMMS_DIR`` at the module pinned from
``git show HEAD:comms-lane/nous_ab.py``: guards go RED there, controls stay
GREEN on both trees. Guards assert the SENTENCE, never a new symbol, so none
can fail pre-fix merely by ``AttributeError`` -- that would bound no behaviour.

    LLM_TOOLKIT_COMMS_DIR=C:/testingtemp/llmstack-1203/prefix \
        python -m pytest tests/test_nous_ab_lane_arrivals.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMS = Path(os.environ.get("LLM_TOOLKIT_COMMS_DIR") or (REPO / "comms-lane"))
if str(COMMS) not in sys.path:
    sys.path.insert(0, str(COMMS))

import nous_ab  # noqa: E402

HOSTED = "http://127.0.0.1:8099/v1/chat/completions"
LOCAL = "http://192.168.1.232:8081/v1/chat/completions"

#: The sentence the pre-fix tree prints for every sweep, arrivals or not.
PRE_FIX_HOSTED = "went to a HOSTED lane"
PRE_FIX_ZERO = "all 0 run(s)"


def _answer_record():
    return {"status": "ANSWER", "finish": "stop", "wall_s": 1.0,
            "content_chars": 900, "reasoning_chars": 0,
            "prompt_tokens": 400, "reasoning_tokens": None,
            "completion_tokens": 300, "wasted_completion_tokens": 0,
            "head": "a substantive answer " * 20}


def _error_record(msg="Connection refused"):
    return {"status": "ERROR", "wall_s": 0.0, "error": msg}


def _run(monkeypatch, tmp_path, capfd, endpoint, records, tasks=None):
    """Drive main() through its real loop with only the dispatch replaced.

    `records` is CYCLED across the runs, so a two-record sequence over a 4-run
    sweep alternates -- the mixed shape this file needs. Repeating the last
    record instead would have made [answer, error] read as 1 arrival of 4, which
    is a property of the harness and not of the summary. (First draft repeated
    the last record and this test failed on 1 == 2: my bug, found by the arm.)
    """
    seq = list(records)
    state = {"i": 0}

    def fake_call(_text, out_path):
        i = state["i"]
        state["i"] += 1
        rec = dict(seq[i % len(seq)])
        Path(out_path).write_text(json.dumps(rec), encoding="utf-8")
        return rec

    def no_shaper(*_a, **_kw):
        raise RuntimeError("shaper unreachable")

    monkeypatch.setattr(nous_ab, "call", fake_call)
    monkeypatch.setattr(nous_ab.shape, "shape_with_usage", no_shaper)
    monkeypatch.setattr(nous_ab, "ENDPOINT", endpoint)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    if tasks is not None:
        monkeypatch.setattr(nous_ab, "TASKS", tasks)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out, out


# ------------------------------------------------------------------ guards ---

def test_an_all_error_sweep_makes_no_lane_claim(monkeypatch, tmp_path, capfd):
    """GUARD: 4 runs, none reached the endpoint -- and the lane was named anyway.

    Pre-fix this printed the byte-identical hosted sentence a working sweep
    prints, so a dead proxy read as a measured placement with 0 local GPUs.
    """
    printed, _out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_error_record()])
    assert PRE_FIX_HOSTED not in printed, (
        "a sweep in which no run reached the endpoint still named a lane; the "
        "placement sentence must not cover runs that never arrived:\n" + printed
    )
    assert "0 of 4" in printed or "NOTHING WAS DISPATCHED" in printed, (
        "the summary did not say that nothing answered:\n" + printed
    )


def test_a_zero_run_sweep_makes_no_lane_claim(monkeypatch, tmp_path, capfd):
    """GUARD: "all 0 run(s) went to a HOSTED lane" is a claim about nothing."""
    printed, _out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_answer_record()], tasks=[])
    assert PRE_FIX_ZERO not in printed, (
        "a sweep that dispatched nothing still stated a lane over 0 runs:\n"
        + printed
    )
    assert "NOTHING WAS DISPATCHED" in printed, (
        "an empty sweep must say that it dispatched nothing:\n" + printed
    )


def test_a_mixed_sweep_says_how_many_arrived(monkeypatch, tmp_path, capfd):
    """GUARD: 2 of 4 answered -- the lane line said "all 4"."""
    printed, _out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_answer_record(), _answer_record(),
                          _error_record(), _error_record()])
    assert "2/4" in printed, (
        "the summary never said how many runs the lane sentence covered:\n"
        + printed
    )
    assert "all 4 run(s)" not in printed, (
        "only 2 of 4 runs answered and the summary said all 4:\n" + printed
    )


def test_the_run_file_records_the_arrival_split(monkeypatch, tmp_path, capfd):
    """GUARD: the console and the saved evidence must not disagree.

    Pre-fix the file had no such key at all, so a reader re-opening it could
    not recover the distinction the console had already collapsed.
    """
    _printed, out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_answer_record(), _error_record()])
    doc = json.loads(out.read_text(encoding="utf-8"))
    split = doc.get("lane_arrivals")
    assert isinstance(split, dict), (
        "the run file records no arrival split, so the placement line cannot "
        "be re-checked after the fact: keys %r" % sorted(doc))
    assert split["dispatched"] == 4, split
    assert split["reached"] == 2, split
    assert split["never_reached"] == 2, split


def test_causes_are_named_per_run(monkeypatch, tmp_path, capfd):
    """GUARD: "N failed" is the same fold one layer up.

    A refused connection, a timeout and a record with no status are three
    problems with three fixes; the summary has to name which one happened.
    The SENTENCE is asserted before the record, so this arm fails the pre-fix
    tree on what it printed rather than on a key that does not exist there.
    """
    printed, out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                        [_error_record("Connection refused"),
                         _error_record("timed out after 180s")])
    assert "Connection refused" in printed, (
        "the cause of a failed run never reached the console: " + printed)
    assert "timed out" in printed, printed
    causes = (json.loads(out.read_text(encoding="utf-8"))
              .get("lane_arrivals") or {}).get("never_reached_causes") or []
    assert any("Connection refused" in c for c in causes), causes
    assert any("timed out" in c for c in causes), causes

def test_a_refused_run_counts_as_an_arrival(monkeypatch, tmp_path, capfd):
    """GUARD: the recorded split must count a refusal as an arrival.

    The over-correction risk of the fix: if "reached" meant "answered with
    content", a model that refused the task or was cut off by its budget would
    be reported as a run that never reached the endpoint. The split is recorded
    so this can be re-checked after the fact, which the pre-fix file cannot.
    """
    _printed, out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_answer_record(),
                          dict(_answer_record(), status="REFUSE"),
                          dict(_answer_record(), status="TRUNCATED")])
    split = (json.loads(out.read_text(encoding="utf-8"))
             .get("lane_arrivals") or {})
    assert split, "the run file records no arrival split"
    assert split["never_reached"] == 0, split
    assert split["reached"] == 4, split


# ---------------------------------------------------------------- controls ---
# These pass on the pre-fix tree too. A control that fails pre-fix is a guard
# mislabelled; a control that can only pass post-fix bounds nothing.

def test_a_successful_hosted_sweep_still_says_hosted(monkeypatch, tmp_path,
                                                     capfd):
    """CONTROL: the fix must not silence the line it corrects."""
    printed, _out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_answer_record()])
    assert PRE_FIX_HOSTED in printed, printed
    assert "0 local" in printed.lower(), printed


def test_a_successful_local_sweep_still_says_local(monkeypatch, tmp_path,
                                                  capfd):
    """CONTROL: the anti-vacuity arm -- the line must still be able to differ."""
    printed, _out = _run(monkeypatch, tmp_path, capfd, LOCAL,
                         [_answer_record()])
    assert "LOCAL lane" in printed, printed
    assert "0 local GPU(s) used" not in printed, printed


def test_the_summary_still_names_model_endpoint_and_writes_the_file(
        monkeypatch, tmp_path, capfd):
    """CONTROL: nothing that already worked may be displaced."""
    printed, out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                        [_answer_record()])
    assert nous_ab.MODEL in printed, printed
    assert "SUMMARY" in printed, printed
    assert HOSTED in printed, printed
    assert "wrote" in printed, printed
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["endpoint"] == HOSTED
    assert doc["model"] == nous_ab.MODEL
    assert doc["runs"]


def test_a_refused_or_truncated_run_is_not_called_unreached(monkeypatch,
                                                            tmp_path, capfd):
    """CONTROL: passes on BOTH trees. An answer is not the only arrival.

    Placement is about the request reaching the lane. A model that REFUSED the
    task, or was cut off by its budget, still placed the work, so the summary
    must not say those runs never reached the endpoint -- folding them into
    "unreached" would over-correct into a different false sentence. Pre-fix
    this held by accident (every run was counted), which is why it is a control
    and not a guard; the recorded count that makes it non-vacuous is pinned by
    test_a_refused_run_counts_as_an_arrival above.
    """
    printed, _out = _run(monkeypatch, tmp_path, capfd, HOSTED,
                         [_answer_record(),
                          dict(_answer_record(), status="REFUSE"),
                          dict(_answer_record(), status="TRUNCATED")])
    assert "NEVER REACHED" not in printed, printed
    assert PRE_FIX_HOSTED in printed, printed


# ------------------------------------------------------------------- pinned --
# The pre-fix module must be the one these guards run against when the
# negative-control env var is set; a green run against the live tree would
# measure nothing at all.

def test_the_negative_control_is_pointed_somewhere():
    """Guard-of-the-guards: name the module under test in the failure output."""
    assert (COMMS / "nous_ab.py").exists(), (
        "no nous_ab.py under %s -- every arm above measured nothing" % (COMMS,))
    override = os.environ.get("LLM_TOOLKIT_COMMS_DIR")
    if override and not Path(override).exists():
        pytest.fail("the pinned comms dir does not exist: %s" % (override,))
