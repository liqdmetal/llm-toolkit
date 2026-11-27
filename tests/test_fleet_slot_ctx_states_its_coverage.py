#!/usr/bin/env python3
"""A minimum over one slot and a minimum over four are different facts.

`fleet_probe`'s spare-capacity line exists to stop work falling through to a
BILLED lane because a local one looked busy. Its previous repair added the
context figure to that certificate, because the only measurably-free lane in the
fleet was a 4,096-token server while `shape.py`'s default shaper model wants
262,144. That repair fixed the missing FIGURE and left the figure's SCOPE
unstated.

MEASURED on 56b01b1, driving `read_slot_state` over five `/slots` documents:

    arm                                          slot_ctx   rendered suffix
    A  all 4 slots state 4096                       4096    ', n_ctx 4096'
    B  all 4 state it, mixed 256k + 4096            4096    ', n_ctx 4096'
    C  ONE slot states 4096, three state nothing    4096    ', n_ctx 4096'
    D  ONE slot states 262144, three nothing      262144    ', n_ctx 262144'
    E  no slot states n_ctx                         None    ''

C and A render BYTE-IDENTICALLY. A router reading "MEASURED spare capacity:
3090-defiant 0/4, n_ctx 4096" on arm C has been told a real measurement and not
how much of the server it covers. llama.cpp assigns a request to whichever slot
is free, so the context a caller can rely on is the least ANY slot offers -- and
a minimum over one of four slots is not a claim about the server.

This is the same defect one field over from the one that wrote the figure in:
a real number reported without the fact that decides whether it can be used.
`counter_only_pids` in cutlass-probe did it with process rows; this does it with
slot context.

WHAT THE FIX DOES
-----------------
`read_slot_state` returns the count of slots that stated a context alongside the
minimum, `probe()` carries it as `slot_ctx_slots`, and `_ctx_suffix` states it
ONLY when it is short of the slot total. A fully-covered server renders exactly
as it did before, because that reading is complete and decorating the common
case would be noise. Arm E still invents nothing: no read, no figure, no count.

GUARDS FAIL BY VALUE ON THE PRE-FIX TREE, NOT BY SYMBOL
-------------------------------------------------------
The arms that read the new count through `read_slot_state` unpack tolerantly
(`res[2] if len(res) > 2 else None`), so pre-fix they land on `None != 1`
instead of a ValueError. The render arms assert the exact string, so pre-fix
they fail on ', n_ctx 4096' where a coverage clause was required.

WHAT IS A CONTROL HERE
----------------------
Four arms must pass on BOTH trees, and they are the counter-arms that keep the
fix from over-reporting: a fully-covered server must render byte-identically, an
unstated context must stay silent, the `(total, busy)` contract must not move,
and `read_slots`'s two-value shorthand must not change shape for its dozen
callers. A guard that only checked "partial differs from full" would reward
decorating every line.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

MODULE_PATH = Path(fleet_probe.__file__).resolve()

#: `shape.py`'s default shaper context, restated so the roomy arm means something.
POLICY_CTX = 262_144


def _slot(n_ctx=None, busy=False):
    d = {"is_processing": busy, "is_processing_prompt": False}
    if n_ctx is not None:
        d["n_ctx"] = n_ctx
    return d


def _covered(result):
    """The covering-slot count from a `read_slot_state` result, on either tree.

    Written tolerantly ON PURPOSE: on the pre-fix tree this returns None instead
    of raising, so the guard fails by VALUE. A control that dies on a tuple
    unpack proves a symbol is missing, not that the behaviour is absent.
    """
    return result[2] if len(result) > 2 else None


class _LlamaCpp(BaseHTTPRequestHandler):
    """A llama.cpp-shaped server. Mutated on the class between tests."""

    health_status = 200
    slots = [{"is_processing": False, "n_ctx": 4096}] * 4
    slots_status = 200

    def do_GET(self):  # noqa: N802 - http.server API
        cls = type(self)
        if self.path == "/health":
            code = cls.health_status
            body = json.dumps({"status": "ok"} if code == 200
                              else {"error": {"message": "Loading model",
                                              "code": 503}})
        elif self.path == "/slots":
            code = cls.slots_status
            body = (json.dumps(cls.slots) if code == 200
                    else json.dumps({"error": {"message": "Invalid API Key",
                                               "code": 401}}))
        elif self.path == "/v1/models":
            code = 401
            body = json.dumps({"error": {"message": "Invalid API Key",
                                         "code": 401}})
        else:
            code, body = 404, json.dumps({"error": "not found"})
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def llama():
    """A fresh llama.cpp server, reset to the measured 4k shape per test."""
    _LlamaCpp.health_status = 200
    _LlamaCpp.slots = [{"is_processing": False, "n_ctx": 4096}] * 4
    _LlamaCpp.slots_status = 200
    srv = HTTPServer(("127.0.0.1", 0), _LlamaCpp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _render(monkeypatch, lanes):
    """Run main() over a synthetic fleet and return what it printed."""
    rows = fleet_probe.probe_all(lanes=lanes)
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    buf = []

    class _Out:
        def write(self, text):
            buf.append(text)

        def flush(self):
            pass

    real = sys.stdout
    sys.stdout = _Out()
    try:
        rc = fleet_probe.main([])
    finally:
        sys.stdout = real
    assert rc == 0, "main() exited %r; a partial server is data, not an error" % rc
    return "".join(buf)


def _suffix(row):
    """`_ctx_suffix` over a caller-built row, on either tree."""
    return fleet_probe._ctx_suffix(row)


# ================================================================== GUARDS


def test_one_measured_slot_is_counted_as_one():
    """THE GUARD on the fact the tuple could not carry."""
    doc = [_slot(4096), _slot(), _slot(), _slot()]
    real = fleet_probe._get
    try:
        fleet_probe._get = lambda url=None, timeout=None, key=None, _d=doc: _d
        res = fleet_probe.read_slot_state("http://127.0.0.1:1", "llamacpp")
    finally:
        fleet_probe._get = real
    assert _covered(res) == 1, (
        "ONE slot stated n_ctx and the reader did not report that only one did; "
        "coverage=%r on %r" % (_covered(res), res))


def test_four_measured_slots_are_counted_as_four():
    """THE GUARD on the other end of the same count."""
    doc = [_slot(4096)] * 4
    real = fleet_probe._get
    try:
        fleet_probe._get = lambda url=None, timeout=None, key=None, _d=doc: _d
        res = fleet_probe.read_slot_state("http://127.0.0.1:1", "llamacpp")
    finally:
        fleet_probe._get = real
    assert _covered(res) == 4, (
        "four slots stated n_ctx and coverage read %r" % (_covered(res),))


def test_no_stated_context_is_counted_as_zero_not_none():
    """THE GUARD that 'measured nothing' is a count, not a silence.

    Arm E stated no context at all. Coverage must be 0 -- a measured zero -- so
    a consumer can tell it apart from a read that never happened.
    """
    doc = [_slot()] * 4
    real = fleet_probe._get
    try:
        fleet_probe._get = lambda url=None, timeout=None, key=None, _d=doc: _d
        res = fleet_probe.read_slot_state("http://127.0.0.1:1", "llamacpp")
    finally:
        fleet_probe._get = real
    assert _covered(res) == 0, (
        "a whole server that stated no n_ctx must count as 0 covering slots, "
        "not None; got %r" % (_covered(res),))


def test_one_of_four_does_not_render_like_four_of_four():
    """THE COLLAPSE. Arms C and A printed the identical suffix on 56b01b1."""
    partial = _suffix({"slot_ctx": 4096, "slot_ctx_slots": 1, "slots": (4, 0)})
    full = _suffix({"slot_ctx": 4096, "slot_ctx_slots": 4, "slots": (4, 0)})
    assert partial != full, (
        "a server where 1 of 4 slots stated a context rendered identically to "
        "one where all 4 did: %r" % (partial,))
    assert "1 of 4" in partial, (
        "the partial suffix must state how many slots it covers: %r" % (partial,))


def test_a_partial_server_states_its_coverage_on_the_operator_line(
        llama, monkeypatch):
    """THE GUARD on the surface a router actually reads."""
    _LlamaCpp.slots = [{"is_processing": False, "n_ctx": 4096},
                       {"is_processing": False},
                       {"is_processing": False},
                       {"is_processing": False}]
    out = _render(monkeypatch, [("thin", llama, "llamacpp")])
    assert "MEASURED spare capacity" in out, out
    assert "1 of 4 slots stated a context" in out, (
        "the spare-capacity line certified a server whose context was measured "
        "on one slot of four without saying so:\n%s" % out)


def test_probe_carries_the_covering_slot_count(llama):
    """THE GUARD on the programmatic surface, not just the prose."""
    row = fleet_probe.probe("defiant", llama, "llamacpp")
    assert row.get("slot_ctx_slots") == 4, (
        "the row dropped how many slots stated the context, so a router cannot "
        "tell a 1-of-4 reading from a 4-of-4 one: %r"
        % (row.get("slot_ctx_slots"),))


def test_a_roomy_partial_server_is_not_accused_of_being_short(llama, monkeypatch):
    """THE GUARD that guidance about coverage does not become a shortfall claim.

    262,144 tokens from one slot is still a roomy reading; the line must say
    'partial', never 'SHORT'. The fix must not turn a scope fact into a verdict.
    """
    _LlamaCpp.slots = [{"is_processing": False, "n_ctx": POLICY_CTX},
                       {"is_processing": False},
                       {"is_processing": False},
                       {"is_processing": False}]
    out = _render(monkeypatch, [("wide", llama, "llamacpp")])
    assert str(POLICY_CTX) in out, out
    assert "1 of 4 slots stated a context" in out, out
    assert "SHORT" not in out.upper(), (
        "a lone roomy slot was flagged as a shortfall:\n%s" % out)


# =============================================================== CONTROLS


def test_a_fully_covered_server_renders_exactly_as_before():
    """CONTROL: the common case must not gain words.

    Pre-fix this renders ', n_ctx 4096'. Post-fix it must still be that string
    to the byte, or every existing reader of this line sees churn for nothing.
    """
    full = _suffix({"slot_ctx": 4096, "slot_ctx_slots": 4, "slots": (4, 0)})
    assert full == ", n_ctx 4096", (
        "a fully-covered server's suffix changed; expected ', n_ctx 4096', "
        "got %r" % (full,))


def test_a_fully_covered_server_line_is_unchanged(llama, monkeypatch):
    """CONTROL on the operator surface for the same property."""
    out = _render(monkeypatch, [("defiant", llama, "llamacpp")])
    assert "MEASURED spare capacity" in out, out
    assert "n_ctx 4096" in out, out
    assert "slots stated a context" not in out, (
        "a 4-of-4 reading was decorated with a coverage clause:\n%s" % out)


def test_an_unstated_context_still_invents_nothing():
    """CONTROL: ABSENT is not 4096 and not a coverage clause."""
    assert _suffix({"slot_ctx": None, "slot_ctx_slots": 0,
                    "slots": (4, 0)}) == "", "an unstated context rendered text"
    assert _suffix({"slot_ctx": None}) == "", "an unstated context rendered text"


def test_the_slots_contract_is_unchanged(llama):
    """CONTROL: a dozen suites pin `row["slots"] == (4, 0)`."""
    row = fleet_probe.probe("defiant", llama, "llamacpp")
    assert row.get("slots") == (4, 0), row.get("slots")


def test_the_two_value_shorthand_did_not_change_shape(llama):
    """CONTROL: `read_slots` is the shorthand every existing caller uses.

    The new count travels on the fuller function only; if the shorthand started
    returning three values it would silently break its callers.
    """
    got = fleet_probe.read_slots(llama, "llamacpp")
    assert got == (4, 0), (
        "read_slots must keep returning just (total, busy); got %r" % (got,))


def test_a_down_lane_reports_no_coverage_count():
    """CONTROL: no read happened, so there is no count to state."""
    row = fleet_probe.probe("nowhere", "http://127.0.0.1:1", "llamacpp",
                            timeout=0.05)
    assert row.get("reachable") is False, row
    assert row.get("slot_ctx_slots") is None, (
        "a lane that was never read got a coverage count: %r"
        % (row.get("slot_ctx_slots"),))


# ================================================== INSTRUMENT CHECKS


def test_this_file_drives_the_repo_module_not_a_shadow_copy():
    """A loaded module must resolve to the repo file.

    This lane has graded a cached or editable-installed copy before; the shadow
    path trap made a control test the FIXED module while claiming to test the
    pre-fix one.
    """
    assert MODULE_PATH == (Path(__file__).resolve().parents[1]
                           / "comms-lane" / "fleet_probe.py").resolve(), (
        "fleet_probe resolved to %s" % (MODULE_PATH,))


def test_the_coverage_count_tracks_the_same_single_read():
    """The count and the minimum must come from ONE document.

    Probing must never perturb what it measures, and two reads of a live server
    can disagree. A count taken from a second read would be a different server's
    fact.
    """
    doc = [_slot(4096), _slot(), _slot(), _slot()]
    calls = {"n": 0}
    real = fleet_probe._get

    def counting(url=None, timeout=None, key=None):
        calls["n"] += 1
        return doc

    try:
        fleet_probe._get = counting
        res = fleet_probe.read_slot_state("http://127.0.0.1:1", "llamacpp")
    finally:
        fleet_probe._get = real
    assert calls["n"] == 1, (
        "read_slot_state made %d /slots reads; the figure and its coverage must "
        "come from one" % calls["n"])
    assert _covered(res) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
