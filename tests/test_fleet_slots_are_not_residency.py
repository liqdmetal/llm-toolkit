"""A loaded server is not a busy server, and the fleet report could not tell.

THE FINDING, measured 2026-11-28 before this file existed
---------------------------------------------------------
``fleet_probe.py`` reads ``/health`` to answer "is a model loaded" and renders
that as ``serving N``. It mentioned ``/slots`` in exactly one place: a comment.
Measured live on this Runner::

    http://127.0.0.1:8090/slots -> 200, 4 slots, 0 busy

    the probe said:  3090-defiant  UP  1 installed, serving 1, 232 ms

A server holding a model with every slot free is the cheapest capacity in the
fleet -- the model is already resident, so it cannot fail to load -- and it
rendered identically to a server saturated by four concurrent requests. That is
the expensive direction for the goal this tool serves. The tool's own header
names the failure it exists to prevent: a lane that looks wrong sends work to a
BILLED one. Spare local capacity that is indistinguishable from full does that.

WHY THIS IS NOT THE HOST-CAPACITY LINE ONE FIELD OVER
-----------------------------------------------------
``host_capacity`` already refuses to call an idle daemon "capacity", correctly:
MEASURED 2026-11-26, the 3090 answered ``{"models":[]}`` to ``/api/ps`` while
every generation on it died ``cudaMemGetInfo failed (out of memory)`` because
Inkling held the card. So "no daemon is serving" does NOT establish capacity.

``/slots`` answers a stronger and different question: the model IS loaded (that
is what ``serving`` establishes from ``/health``) and this says how many
concurrent slots are free right now. Slots say nothing about whether a model CAN
load; they say how much a LOADED one can take. The two lines coexist and assert
different things.

WHAT A SLOT READ CAN AND CANNOT ANSWER
--------------------------------------
``/slots`` is a state read. It lists in-flight work and starts none, so it obeys
the standing "a probe must not perturb what it measures" rule already pinned for
``/health``.

It is KEY-GATED on some boxes. Measured 2026-11-28 on this Runner::

    http://127.0.0.1:8090/slots      -> 200
    q38 (192.168.1.232:8081)         -> 401
    Inkling tunnel (127.0.0.1:18082) -> 401

A 401 is NOT "0 busy" and NOT "full" -- it is "could not ask". So the read
raises and the caller renders ``slots UNKNOWN``, because silence there reads as
"no spare capacity", and a measured-looking zero on an unread lane is the same
defect this file is about, one field over.

GUARDS FAIL ON THE PRE-FIX TREE BY VALUE, NOT BY A MISSING SYMBOL
-----------------------------------------------------------------
``probe()`` and ``main()`` both exist on HEAD, so every guard executes unchanged
on both trees. Pre-fix the row has no ``slots`` key at all, so guards fail on
the row dict and on the rendered text -- by value, never by AttributeError.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402


# --------------------------------------------------------------------------
# A llama.cpp-shaped server. Configuration is mutated on the class between
# tests (the pattern the existing fleet suites use) rather than through
# type() subclasses, so the handler keeps http.server's own __init__.
# --------------------------------------------------------------------------
class _LlamaCpp(BaseHTTPRequestHandler):
    health_status = 200
    slots = [{"is_processing": False}] * 4
    slots_status = 200
    seen = []

    def do_GET(self):  # noqa: N802 - http.server API
        cls = type(self)
        cls.seen.append(self.path)
        if self.path == "/health":
            code = cls.health_status
            body = json.dumps({"status": "ok"} if code == 200
                              else {"error": {"message": "Loading model",
                                              "code": 503}})
        elif self.path == "/slots":
            code = cls.slots_status
            if code == 200:
                body = json.dumps(cls.slots)
            else:
                body = json.dumps({"error": {"message": "Invalid API Key",
                                             "code": 401}})
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

    def log_message(self, *a):  # keep pytest output clean
        pass


class _Ollama(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append(self.path)
        body = ({"models": [{"name": "m1"}]} if self.path == "/api/tags"
                else {"models": []})
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def llama():
    """A fresh llama.cpp server with defaults, reset after every test."""
    _LlamaCpp.health_status = 200
    _LlamaCpp.slots = [{"is_processing": False}] * 4
    _LlamaCpp.slots_status = 200
    _LlamaCpp.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _LlamaCpp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def ollama():
    _Ollama.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _render(monkeypatch, lanes):
    """Run main() over a synthetic fleet and return what it printed.

    ``probe_all(lanes=LANES)`` binds its default at DEFINITION time, so patching
    the module attribute ``LANES`` leaves the real fleet in place -- the first
    draft of this helper did exactly that and every render arm silently measured
    the live Runner instead of its synthetic lanes. Patch the function.
    """
    monkeypatch.setattr(fleet_probe, "probe_all",
                        lambda *a, **k: [fleet_probe.probe(label, url, kind)
                                         for label, url, kind in lanes])
    out = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: out.append(
                            " ".join(str(x) for x in a)))
    fleet_probe.main([])
    return "\n".join(out)


# --------------------------------------------------------------- THE GUARDS


def test_a_loaded_server_with_free_slots_reports_its_busy_count(llama):
    """THE GUARD. ``serving`` alone hides the cheapest capacity in the fleet."""
    row = fleet_probe.probe("d", llama, "llamacpp")
    assert row.get("slots") == (4, 0), (
        "a loaded server with 4 free slots did not report its busy count: "
        "%r (HEAD has no 'slots' key at all)" % (row.get("slots"),))
    assert row["serving"] is True, "residency is a separate fact and unchanged"


def test_a_saturated_server_is_distinguishable_from_a_free_one(llama):
    """THE GUARD, the other half. Pre-fix both rendered ``serving 1``."""
    free = fleet_probe.probe("free", llama, "llamacpp")
    _LlamaCpp.slots = [{"is_processing": True}] * 4
    busy = fleet_probe.probe("busy", llama, "llamacpp")
    assert free.get("slots") != busy.get("slots"), (
        "a free server and a saturated one produce the same reading: %r vs %r"
        % (free.get("slots"), busy.get("slots")))
    assert busy.get("slots") == (4, 4)


def test_a_partially_busy_server_counts_only_the_working_slots(llama):
    """GUARD. The count must be the busy ones, not the total, not a bool."""
    _LlamaCpp.slots = [
        {"is_processing": True}, {"is_processing": False},
        {"is_processing": True}, {"is_processing": False},
    ]
    row = fleet_probe.probe("p", llama, "llamacpp")
    assert row.get("slots") == (4, 2), row.get("slots")


def test_a_key_gated_slots_read_is_unknown_not_zero_busy(llama):
    """THE GUARD that keeps the fix honest. A 401 is not "0 busy".

    Measured live: q38 and the Inkling tunnel both answer 401 to /slots. Folding
    that to "0 busy" would announce spare capacity on two lanes never read --
    the defect this file is about, one field over.
    """
    _LlamaCpp.slots_status = 401
    row = fleet_probe.probe("g", llama, "llamacpp")
    assert row.get("slots") is None, (
        "a 401 on /slots was turned into a value: %r" % (row.get("slots"),))
    assert row.get("slots_error"), "the refusal was not recorded at all"
    assert "401" in row["slots_error"]


def test_the_summary_names_measured_spare_capacity(monkeypatch, llama):
    """THE GUARD on the line a router would actually read."""
    out = _render(monkeypatch, [("d", llama, "llamacpp")])
    assert "every slot free" in out, (
        "the summary did not name measured spare capacity:\n" + out)
    assert "0/4" in out, out


def test_the_summary_does_not_count_an_unread_lane_as_free(monkeypatch, llama):
    """THE GUARD on the summary's honesty: a 401 lane is named, never counted."""
    _LlamaCpp.slots_status = 401
    out = _render(monkeypatch, [("g", llama, "llamacpp")])
    assert "every slot free" not in out, (
        "an unread lane was announced as measured spare capacity:\n" + out)
    assert "could not be read" in out, out
    assert "not counted as free" in out, out


def test_the_render_states_the_busy_count_beside_serving(monkeypatch, llama):
    """GUARD. The lane line itself must carry the number, not just the summary."""
    _LlamaCpp.slots = [
        {"is_processing": True}, {"is_processing": False},
        {"is_processing": False}, {"is_processing": False},
    ]
    out = _render(monkeypatch, [("d", llama, "llamacpp")])
    assert "1/4 slots busy" in out, out


# ------------------------------------------------------------- THE CONTROLS
# These pass on BOTH trees.


def test_control_ollama_rendering_is_unchanged(monkeypatch, ollama):
    """CONTROL, both trees: an ollama lane is untouched by this change."""
    row = fleet_probe.probe("5090", ollama, "ollama")
    assert row["serving"] is False
    assert row["model_count"] == 1
    out = _render(monkeypatch, [("5090", ollama, "ollama")])
    assert "IDLE 0/1 loaded" in out, out
    # an ollama lane has no /slots notion, so it must not gain the word
    assert "slots" not in out, out


def test_control_a_non_llamacpp_lane_reports_na_not_an_error(ollama):
    """CONTROL, both trees: the question does not apply to ollama, and N/A must
    not be reported as an attempted-and-failed read."""
    row = fleet_probe.probe("o", ollama, "ollama")
    assert row.get("slots") is None
    assert row.get("slots_error") is None, (
        "an ollama lane must report N/A, not an error it never attempted")


def test_control_the_probe_still_never_generates(llama):
    """CONTROL, both trees: /slots is a state read; no generation route is hit."""
    fleet_probe.probe("d", llama, "llamacpp")
    seen = list(_LlamaCpp.seen)
    assert "/slots" in seen, seen
    for forbidden in ("/completion", "/v1/chat/completions", "/v1/completions",
                      "/api/generate"):
        assert not any(forbidden in p for p in seen), forbidden


def test_control_the_host_capacity_warning_is_still_printed(monkeypatch, llama):
    """CONTROL, both trees: the older, weaker claim is not replaced. Slots answer
    "is the loaded model busy"; they never answer "can a listed model load", and
    the 2026-11-26 out-of-memory case still stands."""
    out = _render(monkeypatch, [("d", llama, "llamacpp")])
    assert "capacity NOT established" in out, out


def test_control_a_loading_server_stays_up_but_not_serving(llama):
    """CONTROL, both trees: /health 503 is still "loading", and a server that is
    not serving must not be reported as spare capacity even with free slots."""
    _LlamaCpp.health_status = 503
    row = fleet_probe.probe("l", llama, "llamacpp")
    assert row["reachable"] is True
    assert row["serving"] is False
