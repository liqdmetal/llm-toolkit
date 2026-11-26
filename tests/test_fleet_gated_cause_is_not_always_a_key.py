"""The gated-lane line named one cause for six failures and one knob for two lanes.

WHY THIS EXISTS
---------------
``fleet_probe.py``'s summary carries the one line that tells an operator WHY a
loaded local server's spare capacity was not measured, and the one action that
would turn it into a measurement. It read::

    unnamed = [r["lane"] for r in gated if not r.get("key_source")]
    ...
    if unnamed:
        hints.append("no API key configured; set %s or %s"
                     % (key_env_name(unnamed[0]), LANE_KEY_ENV))

Two defects in four lines, both measured 2026-12-03 at 6571990 by driving the
shipped ``main()`` with ``probe_all`` stubbed (the real summary path, only the
probe replaced).

(1) ONE CAUSE FOR EVERY FAILURE. ``slots_error`` is whatever
    ``"%s: %s" % (type(exc).__name__, exc)`` produced, and the hint never read
    it. Six arms, one gated lane, all six printed the identical sentence::

        401 Unauthorized        -> "no API key configured; set FLEET_LANE_API_KEY_Q38"
        timeout                 -> the same sentence
        500 Internal Server Err -> the same sentence
        403 Forbidden           -> the same sentence
        200 but not a slot list -> the same sentence
        404 Not Found           -> the same sentence

    A key cannot fix a timeout, a 500, a missing endpoint, or a payload of the
    wrong shape. An operator who sets a variable because the report told them
    to, reruns, and gets the identical sentence has been sent to fix a thing
    that was never wrong -- and the report is the only surface that could have
    told them otherwise.

(2) ONE LANE'S KNOB FOR EVERY LANE. ``unnamed[0]`` is the FIRST unnamed lane.
    Measured with q38 and 3090-inkling both answering 401::

        line   : ... q38, 3090-inkling -- no API key configured;
                 set FLEET_LANE_API_KEY_Q38 or FLEET_LANE_API_KEY
        needed : FLEET_LANE_API_KEY_Q38, FLEET_LANE_API_KEY_3090_INKLING

    ``FLEET_LANE_API_KEY_3090_INKLING`` appeared NOWHERE in the report. Setting
    the shared ``FLEET_LANE_API_KEY`` does help the second lane, so the sentence
    is not false -- but it is the weaker hint offered beside the stronger one,
    and the per-lane knob is the thing the operator's own row is asking for.
    The live run this fix came from printed exactly this line over exactly
    these two lanes.

WHAT THE FIX DOES
-----------------
``slots_cause(err)`` classifies the recorded failure text into an auth /
timeout / http / shape / unreachable / unreadable cause. The row carries
``slots_cause`` so ``--json`` states it too and the summary never re-guesses
from prose. The hint groups lanes BY CAUSE: only the ``auth`` bucket offers a
key, and it names EVERY unnamed lane's own knob plus the shared fallback; every
other bucket says plainly that no key fixes it and quotes what the server
actually said.

HOW THE ARMS ARE SPLIT
----------------------
GUARD (fails on the module pinned from the pre-fix sha):
    * TestANonAuthFailureIsNotBlamedOnTheMissingKey  (six arms)
    * TestEveryGatedLaneGetsItsOwnKnobNamed          (two lanes, both 401)
    * TestTheRowCarriesTheCauseItRecorded            (--json surface)

CONTROL (passes on BOTH trees -- what stops the guards being satisfied by
deleting the hint):
    * TestTheRealAuthCaseStillNamesTheKnob           (the fix must not remove
                                                      the hint that was right)
    * TestARefusedKeyIsStillReported                 (the pre-existing behaviour
                                                      the sibling suite pins)

The negative control runs this file against the module pinned through
``LLM_TOOLKIT_COMMS_DIR``; unset, it uses the repo's own ``comms-lane``.

    LLM_TOOLKIT_COMMS_DIR=C:/testingtemp/llmstack-1203/prefix \
        python -m pytest tests/test_fleet_gated_cause_is_not_always_a_key.py -q

TRAP this file avoids: the operator-facing variable names are spelled out here
rather than imported from the module. The pre-fix tree has no ``slots_cause``
constants, so a guard referencing them would die on AttributeError THERE and
bound nothing -- the vacuous-control trap the sibling suite documents in
``test_fleet_refused_key_is_not_silent.py``.
"""

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMS = Path(os.environ.get("LLM_TOOLKIT_COMMS_DIR") or (REPO / "comms-lane"))
if str(COMMS) not in sys.path:
    sys.path.insert(0, str(COMMS))

import fleet_probe  # noqa: E402

assert Path(fleet_probe.__file__).resolve().parent == COMMS.resolve(), (
    "imported the wrong fleet_probe: %s" % fleet_probe.__file__
)

# Spelled out, never imported: see the TRAP note in the module docstring.
SHARED_KEY_VAR = "FLEET_LANE_API_KEY"
PER_LANE_PREFIX = "FLEET_LANE_API_KEY_"

E401 = "HTTPError: HTTP Error 401: Unauthorized"
TIMEOUT = "URLError: <urlopen error timed out>"
E500 = "HTTPError: HTTP Error 500: Internal Server Error"
E404 = "HTTPError: HTTP Error 404: Not Found"
SHAPE = "ValueError: /slots answered dict, not a list"

# Every arm that is NOT an auth refusal. A key cannot fix any of them.
NON_AUTH = {
    "timeout": TIMEOUT,
    "http-500": E500,
    "http-404": E404,
    "wrong-payload-shape": SHAPE,
}


def _row(lane, slots_error, key_source=None):
    """One loaded-but-unreadable lane, in the shape ``probe_all`` returns."""
    return {
        "lane": lane, "url": "http://127.0.0.1:1", "reachable": True,
        "models": None, "model_count": None,
        "resident": ["<loaded; name key-gated>"], "resident_count": 1,
        "serving": True, "resident_error": None,
        "slots": None, "slots_error": slots_error,
        "slot_ctx": None, "key_source": key_source,
        "latency_ms": 1, "error": None,
    }


def _render(monkeypatch, rows):
    """Drive the real summary path, replacing only the probe."""
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    for name in list(os.environ):
        if name.startswith(SHARED_KEY_VAR):
            monkeypatch.delenv(name, raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        fleet_probe.main([])
    return buf.getvalue()


def _gated_line(out):
    """The one line this file is about, joined after terminal folding."""
    joined = " ".join(out.split())
    for sentinel in ("could not be read", "busy count could not"):
        if sentinel in joined:
            start = joined.index("[") if "[" in joined else 0
            return joined[start:]
    return ""


# ---- GUARD -------------------------------------------------------------


class TestANonAuthFailureIsNotBlamedOnTheMissingKey:
    """GUARD: the hint must name the cause the probe actually recorded."""

    @pytest.mark.parametrize("arm,err", sorted(NON_AUTH.items()))
    def test_a_non_auth_failure_is_not_blamed_on_the_missing_key(
            self, monkeypatch, arm, err):
        out = _render(monkeypatch, [_row("q38", err)])
        line = _gated_line(out)

        assert line, "the gated line did not print at all:\n%s" % out
        # The defect, stated as the thing that must not be there: a directive
        # to SET a variable for a failure a variable cannot fix.
        assert "set %s%s" % (PER_LANE_PREFIX, "Q38") not in line, (
            "%s was blamed on a missing API key; the report told the operator "
            "to set a variable that cannot fix it. line=%r" % (arm, line))
        assert "no API key configured" not in line, (
            "%s printed the missing-key sentence. line=%r" % (arm, line))
        # And it must say the opposite, not merely stay quiet.
        assert "no API key fixes this" in line, (
            "%s must state that a key is not the fix. line=%r" % (arm, line))

    def test_the_quote_of_what_the_server_said_survives(self, monkeypatch):
        """Naming the cause must not cost the evidence behind it."""
        out = _render(monkeypatch, [_row("q38", TIMEOUT)])
        line = _gated_line(out)

        assert "did not answer" in line, line
        assert "timed out" in line, (
            "the hint must quote what the probe recorded, not paraphrase it "
            "away: %r" % line)


class TestEveryGatedLaneGetsItsOwnKnobNamed:
    """GUARD: two dark lanes need two knobs, not the first lane's knob twice."""

    def test_every_gated_lane_gets_its_own_knob_named(self, monkeypatch):
        out = _render(monkeypatch, [
            _row("q38", E401),
            _row("3090-inkling", E401),
        ])
        line = _gated_line(out)

        assert "q38" in line and "3090-inkling" in line, (
            "both lanes must still be named: %r" % line)
        assert PER_LANE_PREFIX + "Q38" in line, (
            "q38's own knob was dropped: %r" % line)
        assert PER_LANE_PREFIX + "3090_INKLING" in line, (
            "3090-inkling was named in the lane list but its knob "
            "(%s3090_INKLING) appeared nowhere, so an operator setting what "
            "the line prints leaves this lane dark. line=%r"
            % (PER_LANE_PREFIX, line))

    def test_three_gated_lanes_name_three_knobs(self, monkeypatch):
        """Not a special case for two: the bug was ``[0]``, so prove it is gone."""
        out = _render(monkeypatch, [
            _row("q38", E401),
            _row("3090-inkling", E401),
            _row("3090-defiant", E401),
        ])
        line = _gated_line(out)

        for knob in ("Q38", "3090_INKLING", "3090_DEFIANT"):
            assert PER_LANE_PREFIX + knob in line, (
                "%s%s missing from %r" % (PER_LANE_PREFIX, knob, line))


class TestTheRowCarriesTheCauseItRecorded:
    """GUARD: ``--json`` must state the cause, not just the prose error."""

    def test_the_classifier_separates_the_causes(self):
        assert hasattr(fleet_probe, "slots_cause"), (
            "slots_cause is absent: the summary has no way to tell these apart, "
            "so it can only guess -- which is the defect")
        causes = {fleet_probe.slots_cause(e) for e in
                  (E401, TIMEOUT, E500, SHAPE)}
        assert len(causes) >= 4, (
            "four different failures collapsed to %d cause(s): %r"
            % (len(causes), causes))


# ---- CONTROL -----------------------------------------------------------


class TestTheRealAuthCaseStillNamesTheKnob:
    """CONTROL: the hint that was RIGHT must survive the fix."""

    def test_the_row_still_carries_the_error_it_recorded(self):
        # Filed here, not with the guards: this passes on BOTH trees. It is a
        # non-vacuity arm -- ``probe()`` still produces a row at all, so the
        # guard above is reading a row the shipping code really built -- and a
        # guard that passes pre-fix is a control wearing a guard's name.
        # Port 1 refuses at CONNECT, so the row records the lane-level
        # ``error`` and ``slots_error`` stays None -- that is the outer branch,
        # not a slot read. Assert the row names a failure at all, whichever
        # branch produced it. (My first version asserted ``slots_error`` and
        # failed on the FIXED tree: wrong assertion, not a wrong fix.)
        row = fleet_probe.probe("gated", "http://127.0.0.1:1", "llamacpp",
                                timeout=0.4)
        assert isinstance(row, dict), row
        assert row.get("lane") == "gated", row
        assert row.get("error") or row.get("slots_error"), (
            "probe() recorded no failure at all for an unreachable lane: %r"
            % (row,))

    def test_the_real_auth_case_still_names_the_knob(self, monkeypatch):
        out = _render(monkeypatch, [_row("q38", E401)])
        line = _gated_line(out)

        assert PER_LANE_PREFIX + "Q38" in line, (
            "an auth refusal must still name the per-lane knob: %r" % line)
        assert SHARED_KEY_VAR in line, (
            "the shared fallback must still be offered: %r" % line)


class TestARefusedKeyIsStillReported:
    """CONTROL: pre-existing behaviour the sibling suite pins, unchanged here."""

    def test_a_refused_key_is_still_reported(self, monkeypatch):
        out = _render(monkeypatch, [
            _row("q38", E401, key_source=PER_LANE_PREFIX + "Q38"),
        ])
        line = _gated_line(out)

        assert "refused it" in line, (
            "a key that was SENT and refused must still be reported as such, "
            "not as a missing key: %r" % line)
        assert "no API key configured" not in line, (
            "a configured-but-refused key was reported as never configured: %r"
            % line)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
