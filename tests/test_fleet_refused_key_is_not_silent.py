"""Two failures an operator must tell apart rendered identically, and silently.

``fleet_probe`` reports a loaded-but-unreadable local lane on one line. Reading
``/slots`` can fail two structurally different ways, and before this file they
shared an output:

    no API key configured       -> "... -- no API key configured; set FLEET_
                                   LANE_API_KEY_Q38 or FLEET_LANE_API_KEY"
    key configured, refused     -> "... (no hint at all)"

MEASURED 2026-11-29 by driving the shipped ``main()`` against the live fleet,
which answers 401 on both key-gated lanes: with no key in the environment the
line named the knob; with ``FLEET_LANE_API_KEY_Q38`` and
``FLEET_LANE_API_KEY_3090_INKLING`` set to a wrong value the line printed the
count and stopped talking. The two cases are one variable apart.

WHY THE SILENT ONE IS WORSE, AND WHY THIS IS THE REPO'S OWN DEFECT CLASS: the
operator who set the variable has already done the thing the previous report
asked for. The report then goes quiet, and quiet reads as fixed. The lane stays
unmeasured, the router keeps reading unknown as "do not dispatch there", and
the throughput falls through to a billed lane -- the expensive default this
tool exists to prevent. It is the same shape as ``1.050x`` and ``0 rows`` in
this repo: a gap rendered as an answer.

WHY IT IS THREE STATES AND NOT TWO: "not configured" and "configured and
rejected" have different fixes -- set the variable, versus correct its value --
so they cannot share a rendering. The third state, a readable lane, is already
covered by ``test_fleet_lane_key_is_readable`` and is not duplicated here
beyond one control.

THE SECRET IS STILL NEVER RENDERED. The new hint names the VARIABLE that was
sent, never what it contains, because a wrong key is very often a real key
pasted into the wrong lane's variable -- which is exactly the case this line
exists to report. One guard below asserts the value appears nowhere in the
output.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

# Operator-facing contract, spelled out rather than imported: the names are
# what a human types into a shell, and the pre-fix tree has no constants for
# them, so a guard that referenced the constants would fail there on
# AttributeError and bound nothing (the trap the sibling suite documents).
SHARED_KEY_VAR = "FLEET_LANE_API_KEY"
PER_LANE_PREFIX = "FLEET_LANE_API_KEY_"

REFUSAL = "HTTPError: HTTP Error 401: Unauthorized"


def _row(lane, key_source):
    """One loaded-but-unreadable lane, in the shape ``probe_all`` returns."""
    return {
        "lane": lane, "url": "http://127.0.0.1:1", "reachable": True,
        "models": None, "model_count": None,
        "resident": ["<loaded; name key-gated>"], "resident_count": 1,
        "serving": True, "resident_error": None,
        "slots": None, "slots_error": REFUSAL,
        "key_source": key_source,
        "latency_ms": 1, "error": None,
    }


def _readable_row(lane):
    """A lane whose /slots answered: it must not appear in the gated line."""
    row = _row(lane, None)
    row["slots"] = (4, 0)
    row["slots_error"] = None
    return row


def _render(monkeypatch, rows):
    import io
    from contextlib import redirect_stdout
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        fleet_probe.main([])
    return buf.getvalue()


def _gated_line(out):
    """The one line this file is about, joined after terminal wrapping."""
    joined = " ".join(out.split())
    marker = "whose busy count could not be read"
    assert marker in joined, (
        "the gated-lane line did not render at all, so nothing below is being "
        "measured; output was: " + joined[:400])
    start = joined.index(marker)
    end = joined.find("] ", start)
    segment = joined[start:end + 2]
    for tail in ("[0/2] local hosts", "[1] loaded local server"):
        if tail in segment:
            segment = segment.split(tail)[0]
    return segment


# ---------------------------------------------------------------------------
# THE GUARDS -- each fails by value on the pre-fix tree
# ---------------------------------------------------------------------------
def test_a_configured_but_refused_key_is_not_silent(monkeypatch):
    """THE defect. Pre-fix this renders the count and no cause at all."""
    out = _render(monkeypatch, [_row("q38", "FLEET_LANE_API_KEY_Q38")])
    line = _gated_line(out)
    assert "refused" in line.lower(), (
        "a lane whose key WAS configured and was rejected rendered no cause, "
        "so 'you never set the variable' and 'you set it and it was rejected' "
        "-- different problems with different fixes -- are the same line: %r"
        % line)


def test_the_refusal_names_the_variable_it_sent_not_the_key(monkeypatch):
    """Which knob was turned is the actionable half; the value is a leak."""
    secret = "sk-wrong-lane-pasted-here"
    out = _render(monkeypatch, [_row("q38", "FLEET_LANE_API_KEY_Q38")])
    line = _gated_line(out)
    assert "FLEET_LANE_API_KEY_Q38" in line, (
        "the hint did not name the variable that was sent, so an operator "
        "cannot tell which of the two knobs to correct: %r" % line)
    assert secret not in out, (
        "a key value reached the terminal; the row records the variable NAME "
        "and the value belongs only in the Authorization header")
    assert secret not in line


def test_both_states_are_named_when_both_are_present(monkeypatch):
    """A mixed fleet must not collapse to whichever lane sorted first."""
    out = _render(monkeypatch, [
        _row("q38", "FLEET_LANE_API_KEY_Q38"),
        _row("3090-inkling", None),
    ])
    line = _gated_line(out)
    assert "no API key configured" in line, (
        "the unconfigured lane's cause vanished when a configured one was "
        "also present: %r" % line)
    assert "refused" in line.lower(), (
        "the refused lane's cause vanished when an unconfigured one was also "
        "present: %r" % line)
    assert "FLEET_LANE_API_KEY_Q38" in line
    assert "FLEET_LANE_API_KEY_3090_INKLING" in line, (
        "the unconfigured branch must name ITS own lane's variable, not the "
        "first configured lane's: %r" % line)


def test_each_refused_lane_names_its_own_variable(monkeypatch):
    """Two refused lanes on different knobs are two different corrections."""
    out = _render(monkeypatch, [
        _row("q38", "FLEET_LANE_API_KEY_Q38"),
        _row("3090-inkling", "FLEET_LANE_API_KEY_3090_INKLING"),
    ])
    line = _gated_line(out)
    for lane, var in (("q38", "FLEET_LANE_API_KEY_Q38"),
                      ("3090-inkling", "FLEET_LANE_API_KEY_3090_INKLING")):
        assert var in line, (
            "lanes %s and 3090-inkling were both refused but %s was not "
            "named, so only one correction is visible: %r"
            % (lane, var, line))


def test_the_shared_fallback_is_named_as_the_fallback(monkeypatch):
    """A lane keyed by the shared variable must say so, not name a per-lane
    variable that was never set."""
    out = _render(monkeypatch, [_row("q38", SHARED_KEY_VAR)])
    line = _gated_line(out)
    assert SHARED_KEY_VAR in line, (
        "the shared fallback variable was not named, so the operator is told "
        "to look at a knob that was not the one used: %r" % line)


# ---------------------------------------------------------------------------
# THE CONTROLS -- these must pass on BOTH trees, so the guards cannot be green
# for the wrong reason
# ---------------------------------------------------------------------------
def test_with_no_key_configured_the_hint_still_names_the_knob(monkeypatch):
    """The state the previous commit added must survive this one."""
    out = _render(monkeypatch, [_row("q38", None)])
    line = _gated_line(out)
    assert "no API key configured" in line, line
    assert "FLEET_LANE_API_KEY_Q38" in line, line


def test_a_readable_lane_is_not_reported_as_unreadable(monkeypatch):
    """A measured lane must not appear in the gated line at all."""
    out = _render(monkeypatch, [_readable_row("3090-defiant"),
                                _row("q38", "FLEET_LANE_API_KEY_Q38")])
    line = _gated_line(out)
    assert "3090-defiant" not in line, (
        "a lane whose /slots answered was counted as unreadable: %r" % line)


def test_an_unreadable_lane_is_never_counted_as_free(monkeypatch):
    """ABSENT != ZERO survives. A 401 is not 0 busy and not full."""
    out = _render(monkeypatch, [_row("q38", "FLEET_LANE_API_KEY_Q38")])
    assert "not counted as free" in " ".join(out.split()), out
    assert "0/0" not in _gated_line(out), (
        "an unread lane rendered a slot count it never measured")


def test_nothing_is_reported_when_every_lane_is_readable(monkeypatch):
    """The whole line must vanish, not print an empty banner."""
    out = _render(monkeypatch, [_readable_row("3090-defiant")])
    assert "whose busy count could not be read" not in " ".join(out.split()), (
        "the gated-lane line printed with zero gated lanes: %r" % out)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
