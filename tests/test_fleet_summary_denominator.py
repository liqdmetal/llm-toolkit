"""A local lane whose residency probe failed must not vanish from the count.

The defect
----------
``main()`` builds its headline number from the rows whose RESIDENCY ANSWERED::

    local = [r for r in rows if r["resident"] is not None]
    idle  = [r for r in local if not r["serving"]]
    print("[%d/%d] local lanes SERVING (idle: %s)" ...)

``resident`` is ``None`` in two different situations that must not be confused:

    hosted lane (nous)               resident None, resident_error None
    local lane whose /api/ps FAILED  resident None, resident_error "<exc>"

Both are filtered out by the same test, so a local lane that answered
``/api/tags`` and then failed ``/api/ps`` is not counted at all. The printed
denominator shrinks to the lanes that happened to report residency.

MEASURED 2026-11-25 with three rows built to that shape (3090 reachable, 7
models installed, ``resident_error = "URLError: 404"``)::

    local lanes counted: 1  -> ['5090']
    the line:  [0/1] local lanes SERVING (idle: 5090)

Two local lanes exist. The line says one, and says it with the same confidence
it would have if the 3090 had never been in the fleet. That is the defect this
repo already fixed twice elsewhere -- ``5d0d05a`` (a shaper with the wrong model
must skip, not break the caller) and the residency read itself in ``ea80b99`` --
in its third form: ABSENT and EMPTY collapsed in the DENOMINATOR rather than in
a status field.

Why the denominator is the expensive place for it
-------------------------------------------------
This number is the reason the tool exists: "a reachable-but-idle lane is the
expensive default, because the router falls through to a billed one". A lane
that drops out of the denominator makes the fleet look SMALLER AND HEALTHIER than
it is. ``[0/1]`` is a statement that every local lane was accounted for and all
of them are idle; the truth was that one local lane could not be asked.

What is asserted
----------------
The RENDERED LINE, via ``main()`` with ``probe_all`` stubbed. Not a new helper's
return value: a control that asserts on a dict key the fix introduces fails on
HEAD by ``KeyError`` and bounds nothing. The line is the contract.

Controls, so the fix cannot be "always print zeros":
  - a fully healthy fleet still reports its idle/serving split correctly
  - a hosted-only fleet invents no local line at all
  - a local lane that really is idle is still named as idle, not as unknown
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402


def _row(lane, *, hosted=False, resident=(), resident_error=None,
         model_count=3, reachable=True, error=None):
    """One probe row in exactly the shape ``probe()`` returns."""
    if hosted:
        resident_list, resident_count, serving = None, None, None
    elif resident_error is not None:
        resident_list, resident_count, serving = None, None, None
    else:
        resident_list = list(resident)
        resident_count = len(resident_list)
        serving = bool(resident_list)
    return {
        "lane": lane, "url": "http://x", "reachable": reachable,
        "models": ["m"] * model_count, "model_count": model_count,
        "resident": resident_list, "resident_count": resident_count,
        "serving": serving, "resident_error": resident_error,
        "latency_ms": 1, "error": error,
    }


def _summary(rows, monkeypatch):
    """Run main() against these rows and return what it printed."""
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fleet_probe.main([])
    assert rc == 0, "a down lane is data; main() must still exit 0"
    return buf.getvalue()


def _headline(out):
    """The `[n/m] local lanes SERVING ...` line, or None if absent."""
    for line in out.splitlines():
        if "local lanes SERVING" in line:
            return line
    return None


def test_a_local_lane_whose_residency_failed_stays_in_the_denominator(monkeypatch):
    """THE GUARD. Two local lanes exist; the line must not say one."""
    rows = [
        _row("nous", hosted=True),
        _row("5090", resident=[], model_count=2),
        _row("3090", resident_error="URLError: 404", model_count=7),
    ]
    line = _headline(_summary(rows, monkeypatch))
    assert line is not None, "main() printed no local-lane summary at all"
    assert "[0/2]" in line, (
        "two local lanes are in the fleet but the summary counted one: %r -- a "
        "lane whose /api/ps failed was dropped from the denominator, so the "
        "fleet reads smaller and healthier than it is" % (line,))


def test_a_local_lane_whose_residency_failed_is_named_as_unknown(monkeypatch):
    """Dropping it from the denominator is half the defect; the other half is
    that nothing on the line says its residency could not be read."""
    rows = [
        _row("nous", hosted=True),
        _row("5090", resident=[], model_count=2),
        _row("3090", resident_error="URLError: 404", model_count=7),
    ]
    out = _summary(rows, monkeypatch)
    assert "3090" in out, (
        "the 3090 does not appear anywhere in the report, so an operator is "
        "told nothing about a lane that is up with 7 models installed")
    # The per-lane detail line already says why; the summary must not contradict
    # it by implying every local lane was accounted for.
    assert "404" in out, "the reason the residency read failed is not in the output"


def test_control_a_healthy_fleet_still_reports_its_split(monkeypatch):
    """Both controls below pass on HEAD too; a fix that broke them is a new bug."""
    rows = [
        _row("nous", hosted=True),
        _row("5090", resident=["a"], model_count=2),
        _row("3090", resident=[], model_count=7),
    ]
    line = _headline(_summary(rows, monkeypatch))
    assert line is not None
    assert "[1/2]" in line, "one of two local lanes is serving, but: %r" % (line,)
    assert "3090" in line, "the idle lane must be named as idle: %r" % (line,)


def test_control_a_hosted_only_fleet_invents_no_local_line(monkeypatch):
    """No local lanes means no local line -- not `[0/0]` and not a fake one."""
    rows = [_row("nous", hosted=True)]
    out = _summary(rows, monkeypatch)
    assert _headline(out) is None, (
        "a hosted-only fleet printed a local-lane summary: %r" % (out,))
    assert "[1/1] lanes reachable" in out


def test_an_all_failed_residency_fleet_is_not_a_healthy_zero(monkeypatch):
    """The extreme case, and a GUARD not a control: it also fails on HEAD.

    `[0/0]` is the shape a reader skims past. If the denominator is built only
    from lanes that answered, a fleet of two unreadable local lanes renders as
    a fleet with no local capacity at all -- and on HEAD the summary line
    disappears entirely, which reads as "this box has no local lanes".
    """
    rows = [
        _row("nous", hosted=True),
        _row("5090", resident_error="TimeoutError: timed out", model_count=3),
        _row("3090", resident_error="URLError: 404", model_count=7),
    ]
    line = _headline(_summary(rows, monkeypatch))
    assert line is not None, (
        "both local lanes failed residency and the summary line disappeared "
        "entirely, which reads as 'no local capacity'")
    assert "[0/2]" in line, (
        "two local lanes exist and neither could be read; the line must say so, "
        "not drop them: %r" % (line,))
