"""A probe that never generates cannot establish CAPACITY.

The defect this pins was measured against the live fleet on 2026-11-26:

    3090 /api/ps           -> {"models":[]}          (ollama idle)
    3090 /api/generate     -> HTTP 500
                              cudaMemGetInfo failed (out of memory)
                              cudaMalloc failed
    fleet_probe.py         -> [1/2] local hosts with idle capacity

Every part of that box's idleness reading was true and the conclusion drawn
from it was not. Nothing was serving, and nothing COULD serve: inkling held
the 3090's VRAM, so the shaper model that /api/tags lists could not be
loaded. The probe never starts a generation -- deliberately, so it cannot
evict a resident model -- which means the strongest fact it can establish is
"no daemon is serving". Capacity is a different claim and it needs a
generation to support it.

Why this direction is the expensive one: this tool exists to stop the
fall-through to a billed lane. A line reading "idle capacity" invites a
dispatch onto a host that answers every listing question and then fails to
load, so the work comes back empty and the router does what the tool was
built to prevent. That is the same shape as the ABSENT/EMPTY confusion this
repo has fixed three times (ea80b99, 5d0d05a, and the machine-vs-daemon count
in test_fleet_host_capacity.py) -- one level further out, in the conclusion
rather than in the field.

What is asserted
----------------
The RENDERED LINE, via main() with probe_all stubbed -- the contract every
other test in this directory uses. Not a helper's return value: asserting on
something the fix introduces fails on HEAD by NameError/AttributeError and
bounds nothing.

Controls, so the fix cannot decay into "never say anything useful":
  - two genuinely separate idle hosts are still both counted
  - a daemon that IS serving is never described as not serving
  - a hosted-only fleet still invents no local-host line
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

H5090 = "http://192.168.1.232:11434"
HQ38 = "http://192.168.1.232:8081"
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

def test_the_capacity_line_does_not_claim_capacity(monkeypatch):
    """THE GUARD. The exact live shape: 3090 answers /api/ps empty and cannot
    load. A listing-only probe must not promote that to spare capacity."""
    rows = [
        _row("nous", "http://127.0.0.1:8099", hosted=True),
        _row("5090", H5090, resident=["a"], model_count=3),
        _row("3090", H3090, resident=[], model_count=7),
    ]
    line = _host_line(_run(rows, monkeypatch))
    assert line is not None, "no local-host line was printed"
    assert "with idle capacity" not in line, (
        "the probe never starts a generation, so it cannot establish "
        "capacity; a listed model can still fail to load: %r" % (line,))
    assert "no daemon serving" in line, (
        "the line must state the fact it did measure: %r" % (line,))


def test_the_capacity_line_states_its_own_limit(monkeypatch):
    """The count alone is what mislead. The limit has to travel with it, or an
    operator reading only the bracket has no reason to doubt it."""
    rows = [_row("3090", H3090, resident=[], model_count=7)]
    line = _host_line(_run(rows, monkeypatch))
    assert line is not None
    low = line.lower()
    assert "not established" in low or "cannot" in low or "limit" in low, (
        "the line states a count but not what the count does not prove: %r"
        % (line,))


# ---------------------------------------------------------------- controls ---

def test_control_two_separate_idle_hosts_are_still_both_counted(monkeypatch):
    """Passes on HEAD too. A fix that printed [0/0], or dropped the line
    whenever any host was idle, would fail here -- the count must survive."""
    rows = [
        _row("a", "http://10.0.0.1:11434", resident=[], model_count=2),
        _row("b", "http://10.0.0.2:11434", resident=[], model_count=5),
    ]
    line = _host_line(_run(rows, monkeypatch))
    assert line is not None, (
        "two idle machines on two hosts must still produce a host line")
    assert "[2/2]" in line, (
        "both hosts have no daemon serving and both must be counted: %r"
        % (line,))


def test_control_a_serving_daemon_is_never_called_not_serving(monkeypatch):
    """The opposite error: a host with a loaded model must not be counted as
    free just because this line's wording changed."""
    rows = [
        _row("5090", H5090, resident=["a"], model_count=3),
        _row("3090", H3090, resident=[], model_count=7),
    ]
    line = _host_line(_run(rows, monkeypatch))
    assert line is not None
    assert "[1/2]" in line, (
        "one host is serving and only the other is free: %r" % (line,))


def test_control_a_hosted_only_fleet_invents_no_host_line(monkeypatch):
    rows = [_row("nous", "http://127.0.0.1:8099", hosted=True)]
    out = _run(rows, monkeypatch)
    assert _host_line(out) is None, (
        "a hosted-only fleet printed a local-machine line: %r" % (out,))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
