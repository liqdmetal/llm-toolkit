""""MEASURED spare capacity" without the context length is a claim no router can use.

THE FINDING, measured 2026-12-02 before this file existed
---------------------------------------------------------
`fleet_probe.py` reads `/slots` to answer the one question the whole tool exists
for: which local lane can take work RIGHT NOW. That read returns a list whose
every element carries `n_ctx` -- the usable context of that slot. `read_slots`
keeps `len(doc)` and the busy count and throws `n_ctx` away, and the summary
prints:

    [1] loaded local server(s) with every slot free (MEASURED spare capacity:
        3090-defiant 0/4)

MEASURED live this run, `http://127.0.0.1:8090/slots`:

    id=0 n_ctx=4096 is_processing=False
    id=1 n_ctx=4096 is_processing=False
    id=2 n_ctx=4096 is_processing=False
    id=3 n_ctx=4096 is_processing=False

So the ONE lane this report certifies as spare capacity -- the only lane in the
fleet whose busy count is even readable -- is a **4,096-token** server, while
`shape.py`'s default shaper model is `defiant9b-abliterated:256k`, i.e. 262,144
tokens. A router that trusts "MEASURED spare capacity: 3090-defiant 0/4" and
sends it a 20k-token prompt gets a context-exceeded failure from a box the tool
had just certified as free.

WHY THIS IS THE EXPENSIVE DIRECTION, WHICH IS THIS TOOL'S OWN STANDARD
---------------------------------------------------------------------
The probe's stated purpose is to stop work falling through to a BILLED lane
because a local one looked busy. This defect inverts it: the local lane looks
free and cannot hold the work, so the request is made, fails, and falls through
to the frontier lane anyway -- with the probe's own certificate as the reason it
was tried. The number was not wrong. The number was INCOMPLETE in the same way
`counter_only_pids` was in cutlass-probe: a real measurement reported without
the coverage fact that decides whether it can be used.

THE GUARDS FAIL ON THE PRE-FIX TREE BY VALUE
--------------------------------------------
`/slots` is stubbed with a real llama.cpp-shaped server on a loopback port, so no
outside lane is touched and no generation is started. Pre-fix the row has no
`slot_ctx` key at all and the summary line carries no context figure, so every
guard fails on a value that is missing rather than on a missing symbol.

WHAT IS A CONTROL HERE AND WHAT IS NOT (corrected mid-run)
----------------------------------------------------------
I first labelled five arms "controls" and claimed they all pass on both trees.
Measured pre-fix, only THREE do: a server stating no `n_ctx` gets none, the
`(total, busy)` contract is unchanged, and a refused read yields no context.
The other two -- "a roomy server is not reported as short" and "a busy slot
still counts busy alongside the context" -- assert the new fact, so they can
only pass after the fix. They are GUARDS, and they are the useful direction of
one: they fail if the repair over-flags a correct lane or confuses the busy
count with the context. The docstring says guard for them rather than calling
them controls, because a control that only passes after the fix is a guard
wearing a control's name.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

#: The context the policy's own default shaper model carries (`shape.py`,
#: `defiant9b-abliterated:256k`). Stated here so the guard can say what "too
#: small" means instead of asserting a bare inequality that any number satisfies.
POLICY_CTX = 262_144


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
    """A fresh llama.cpp server holding the measured 4k shape, reset per test."""
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
    assert rc == 0, "main() exited %r; a down lane is data, not an error" % rc
    return "".join(buf)


# ------------------------------------------------------------------ THE GUARDS


def test_the_slot_context_is_carried_on_the_row(llama):
    """THE GUARD. Pre-fix the row has no context key, so a programmatic router
    cannot see that this certified-free lane holds 4k."""
    row = fleet_probe.probe("defiant", llama, "llamacpp")
    assert row.get("slot_ctx") == 4096, (
        "/slots stated n_ctx=4096 for every slot and the row dropped it, so "
        "a router reading 'spare capacity' cannot tell a 4k server from a "
        "256k one: slot_ctx=%r" % (row.get("slot_ctx"),))


def test_the_spare_capacity_line_states_the_context(llama, monkeypatch):
    """THE GUARD on the operator surface: the number the router reads."""
    out = _render(monkeypatch, [("defiant", llama, "llamacpp")])
    assert "MEASURED spare capacity" in out, out
    assert "4096" in out, (
        "the only lane this run certifies as spare capacity is a 4k server "
        "and the line that certifies it says nothing about context:\n%s"
        % out)


def test_the_context_is_smaller_than_the_policy_model(llama, monkeypatch):
    """THE GUARD on the consequence, not just the wording.

    The point of the figure is that it is 64x short of the policy's default
    shaper model. Pinned as a relationship so the guard cannot be satisfied by
    printing a number that is wrong.
    """
    out = _render(monkeypatch, [("defiant", llama, "llamacpp")])
    row_ctx = fleet_probe.probe("defiant", llama, "llamacpp").get("slot_ctx")
    assert isinstance(row_ctx, int) and 0 < row_ctx < POLICY_CTX, (
        "expected a measured context below the policy model's %d; got %r"
        % (POLICY_CTX, row_ctx))
    assert str(row_ctx) in out, out


# ---------------------------------------------------------------- THE CONTROLS


def test_a_server_that_states_no_context_invents_none(llama, monkeypatch):
    """CONTROL: ABSENT is not 4096 and not 0. A server that does not say must
    not be given a number, or the field becomes a guess on two lanes."""
    _LlamaCpp.slots = [{"is_processing": False}] * 4
    row = fleet_probe.probe("quiet", llama, "llamacpp")
    assert row.get("slot_ctx") is None, (
        "a server stating no n_ctx was given %r" % (row.get("slot_ctx"),))
    out = _render(monkeypatch, [("quiet", llama, "llamacpp")])
    assert "MEASURED spare capacity" in out, out
    assert "n_ctx" not in out, (
        "an unstated context must not be rendered as a figure:\n%s" % out)


def test_a_roomy_server_is_not_reported_as_short(llama, monkeypatch):
    """THE GUARD for the other direction: the fix must not accuse a correct
    lane. 262,144 tokens is the policy model's own context, not a shortfall.

    Not a control: it asserts the new fact, so it cannot pass pre-fix."""
    _LlamaCpp.slots = [{"is_processing": False, "n_ctx": POLICY_CTX}] * 4
    row = fleet_probe.probe("roomy", llama, "llamacpp")
    assert row.get("slot_ctx") == POLICY_CTX, row.get("slot_ctx")
    out = _render(monkeypatch, [("roomy", llama, "llamacpp")])
    assert str(POLICY_CTX) in out, out
    assert "SHORT" not in out.upper(), (
        "a lane holding the policy model's own context was flagged as short:"
        "\n%s" % out)


def test_the_slots_contract_is_unchanged(llama):
    """CONTROL: a dozen existing suites assert `row["slots"] == (4, 0)`. This
    change adds a fact; it may not re-shape the one they pin."""
    row = fleet_probe.probe("defiant", llama, "llamacpp")
    assert row.get("slots") == (4, 0), row.get("slots")


def test_a_busy_slot_still_counts_busy_alongside_the_context(llama):
    """THE GUARD that the busy count and the context are not confused: both are
    read from the same document and one must not be reported as the other.

    The `slots == (4, 1)` half holds on both trees; the `slot_ctx` half is the
    new fact, which is why this arm cannot be a control."""
    _LlamaCpp.slots = [{"is_processing": True, "n_ctx": 4096},
                       {"is_processing": False, "n_ctx": 4096},
                       {"is_processing": False, "n_ctx": 4096},
                       {"is_processing": False, "n_ctx": 4096}]
    row = fleet_probe.probe("part", llama, "llamacpp")
    assert row.get("slots") == (4, 1), row.get("slots")
    assert row.get("slot_ctx") == 4096, row.get("slot_ctx")


def test_a_refused_slots_read_yields_no_context(llama):
    """CONTROL: a 401 is 'could not ask', never 'no slots' and never a context.
    The key-gated lanes must not gain a fabricated figure."""
    _LlamaCpp.slots_status = 401
    row = fleet_probe.probe("gated", llama, "llamacpp")
    assert row.get("slots") is None, row.get("slots")
    assert row.get("slot_ctx") is None, row.get("slot_ctx")
