"""Idle capacity is counted in MACHINES, not in HTTP daemons.

The defect
----------
``main()`` counts local lanes:

    local = [r for r in rows if r["resident"] is not None or ...]
    idle  = [r for r in local if not r["serving"]]
    print("[%d/%d] local lanes SERVING (idle: %s)" ...)

One lane is one daemon. One host can run several. MEASURED 2026-11-26 against
the real fleet::

    nous  UP    410 installed, hosted, 289 ms
    5090  UP    3 installed, IDLE 0/3 loaded, 13 ms     <- 192.168.1.232:11434
    q38   UP    ? installed, serving 1, 30 ms           <- 192.168.1.232:8081
    3090  UP    7 installed, IDLE 0/7 loaded, 16 ms
    [4/4] lanes reachable
    [1/3] local lanes SERVING (idle: 5090, 3090)

"idle: 5090, 3090" names two idle lanes. Behind them are two machines, and one
of those two is ALREADY SERVING a loaded model on its other daemon. So the line
advertises two units of spare local capacity where one exists, and the one it
overstates is the 5090 -- the GPU that must not be disturbed while it serves.

Why that direction is the expensive one
---------------------------------------
This tool's whole reason for existing is the fall-through to a billed lane: "a
reachable-but-idle lane is the expensive default". A lane count that runs high
does the opposite of a count that runs low -- it reads as *more* room than there
is, and the router dispatches onto a card that is busy. The repo has already
fixed the low-running form twice (``5de9117``-class denominator work in
``ea80b99`` and ``5d0d05a``); this is the same ABSENT/EMPTY confusion one level
up, in the unit of measurement rather than in a status field.

An idle ollama daemon does not establish that the card is free. The evidence
that a host is free must be that NO daemon on it is serving.

What is asserted
----------------
The RENDERED LINE. Not ``host_capacity()``'s return value: a control asserting
on a helper the fix introduces fails on HEAD by ``AttributeError`` and bounds
nothing. ``main()`` with ``probe_all`` stubbed is the contract.

Controls, so the fix cannot be "always say zero hosts":
  - two genuinely separate idle hosts still report 2
  - a hosted-only fleet invents no local-host line at all
  - a host nobody could read is not counted as free, and is named
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

H5090 = "http://192.168.1.232:11434"   # ollama on the 5090 host
HQ38 = "http://192.168.1.232:8081"     # llama.cpp on the SAME host
H3090 = "http://192.168.1.124:11434"


def _row(lane, url, *, hosted=False, resident=(), resident_error=None,
         model_count=3):
    """One probe row in exactly the shape ``probe()`` returns."""
    if hosted or resident_error is not None:
        resident_list, resident_count, serving = None, None, None
    else:
        resident_list = list(resident)
        resident_count = len(resident_list)
        serving = bool(resident_list)
    return {
        "lane": lane, "url": url, "reachable": True,
        "models": None if model_count is None else ["m"] * model_count,
        "model_count": model_count,
        "resident": resident_list, "resident_count": resident_count,
        "serving": serving, "resident_error": resident_error,
        "latency_ms": 1, "error": None,
    }


def _run(rows, monkeypatch):
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fleet_probe.main([])
    assert rc == 0, "a down lane is data; main() must still exit 0"
    return buf.getvalue()


def _host_line(out):
    for line in out.splitlines():
        if "local host" in line:
            return line
    return None


# ------------------------------------------------------------------ guards ---

def test_an_idle_daemon_on_a_serving_host_is_not_idle_capacity(monkeypatch):
    """THE GUARD. The real fleet shape, reduced to its two daemons.

    The 5090's ollama is idle; q38's llama.cpp on the same hostname has a model
    loaded. One machine, and it is busy.
    """
    rows = [
        _row("nous", "http://127.0.0.1:8099", hosted=True),
        _row("5090", H5090, resident=[], model_count=3),
        _row("q38", HQ38, resident=["<loaded; name key-gated>"], model_count=None),
        _row("3090", H3090, resident=[], model_count=7),
    ]
    line = _host_line(_run(rows, monkeypatch))
    assert line is not None, "no local-host capacity line was printed"
    assert "[1/2]" in line, (
        "two local machines exist and one is free (billed fall-through is still "
        "avoidable on exactly one of them), but the line says: %r" % (line,))
    assert "5090" in line, (
        "the line must name the daemon whose idleness is an artifact of which "
        "daemon was asked, or an operator reads 'idle: 5090' as a free card: "
        "%r" % (line,))


def test_the_lane_line_alone_overstates_spare_capacity(monkeypatch):
    """Both lines together: the lane count is not allowed to be the last word.

    This is the reading that matters. `[1/3] local lanes SERVING` with
    `idle: 5090, 3090` looks like two cards of headroom; the machine line must
    be there to correct it down to one.
    """
    rows = [
        _row("5090", H5090, resident=[], model_count=3),
        _row("q38", HQ38, resident=["<loaded>"], model_count=None),
        _row("3090", H3090, resident=[], model_count=7),
    ]
    out = _run(rows, monkeypatch)
    lanes = [l for l in out.splitlines() if "local lanes SERVING" in l]
    hosts = _host_line(out)
    assert lanes and hosts, (
        "the run must state both the lane count and the machine count:\n%s" % out)
    assert "5090, 3090" in lanes[0], "precondition: the lane line still names both"
    assert "[1/2]" in hosts, (
        "the machine count does not correct the lane count: %r" % (hosts,))


# ---------------------------------------------------------------- controls ---

def test_control_two_separate_idle_hosts_are_both_counted(monkeypatch):
    """Passes on HEAD too. A fix that always printed [0/2] would fail here.

    The control asserts the LANE line, which exists on both trees. Asserting
    the host line would make this a guard wearing a control's name -- it fails
    on HEAD because the line is absent, not because a count is wrong.
    """
    rows = [
        _row("a", "http://10.0.0.1:11434", resident=[], model_count=2),
        _row("b", "http://10.0.0.2:11434", resident=[], model_count=5),
    ]
    out = _run(rows, monkeypatch)
    assert "[0/2] local lanes SERVING (idle: a, b)" in out, (
        "two idle machines on two hosts must both be named as idle:\n%s" % out)


def test_control_a_hosted_only_fleet_invents_no_host_line(monkeypatch):
    rows = [_row("nous", "http://127.0.0.1:8099", hosted=True)]
    out = _run(rows, monkeypatch)
    assert _host_line(out) is None, (
        "a hosted-only fleet printed a local-machine line: %r" % (out,))


def test_control_an_unread_host_is_not_counted_as_free(monkeypatch):
    """A machine nobody could ask is unknown capacity, never spare capacity.

    Asserts the pre-existing lane line and the per-lane residency note, both of
    which are present on HEAD -- so this control passes before and after.
    """
    rows = [
        _row("5090", H5090, resident=[], model_count=3),
        _row("3090", H3090, resident_error="URLError: 404", model_count=7),
    ]
    out = _run(rows, monkeypatch)
    assert "[0/2] local lanes SERVING" in out, (
        "neither daemon is serving, and the denominator must hold both:\n%s" % out)
    assert "residency unknown" in out and "404" in out, (
        "the reason the 3090 could not be read must stay in the output:\n%s"
        % out)


def test_control_one_daemon_per_host_still_splits_correctly(monkeypatch):
    """The ordinary fleet: machines and daemons agree, so nothing changes."""
    rows = [
        _row("5090", H5090, resident=["a"], model_count=3),
        _row("3090", H3090, resident=[], model_count=7),
    ]
    out = _run(rows, monkeypatch)
    assert "[1/2] local lanes SERVING" in out, (
        "one serving, one idle, one daemon each:\n%s" % out)
    line = _host_line(out)
    if line is not None:  # only exists after the fix; the control is the lane line
        assert "idle daemons on a busy host" not in line, (
            "no daemon shares a serving host here, so the note must be absent: "
            "%r" % (line,))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
