"""A serving local lane must not be reported as IDLE because of its server kind.

THE FINDING, measured 2026-11-26 before this file existed
---------------------------------------------------------
``fleet_probe.py`` reported::

    5090 UP    3 installed, IDLE 0/3 loaded, 21 ms
    [0/2] local lanes SERVING (idle: 5090, 3090)

while a different inference server on the SAME box answered ready, three times
in a row::

    $ curl -s -D - -o /dev/null http://192.168.1.232:8081/health
    HTTP/1.1 200 OK
    Server: llama.cpp
    {"status":"ok"}

Both statements are true and only one of them is in the report. The probe knows
two lane kinds, ``ollama`` (``/api/tags`` + ``/api/ps``) and ``openai``
(``/v1/models``), and asks each box only that question. The 5090 runs BOTH an
ollama daemon with nothing loaded and a llama.cpp server with a model loaded, so
``/api/ps`` answering ``{"models":[]}`` was read as "this lane has no capacity"
when it meant "the ollama daemon on this box has no capacity".

WHY THIS IS THE EXPENSIVE DEFECT AND NOT A COSMETIC ONE
------------------------------------------------------
The whole reason this tool exists, in its own header: "a reachable-but-idle lane
is the expensive default this tool exists to expose, because the router falls
through to a billed one." A lane reported IDLE while it is serving produces
exactly that fall-through, and the fleet line ``[0/2] local lanes SERVING`` reads
as a statement that every local lane was asked and none had capacity. On the
measurement above, one of them did.

This is the same defect class the repo has fixed twice already, in a third
place: ``ea80b99`` separated ABSENT from EMPTY in a residency field, and
``5de5cf7`` fixed the DENOMINATOR that dropped a lane it could not read. Here a
lane is neither absent nor empty -- it is SERVING, on a kind nobody asked about.

WHAT A llama.cpp LANE CAN AND CANNOT ANSWER
-------------------------------------------
  - ``/health`` is the readiness signal and needs no key: 200 with
    ``{"status":"ok"}`` when a model is loaded, 503 while loading. It is a state
    read and loads nothing, so it satisfies the "a probe must never perturb what
    it measures" rule the ollama paths already follow.
  - ``/v1/models`` is KEY-GATED on this box: it answers 401. A listing that
    FAILED is not an absence, so the model count must be reported UNKNOWN (None)
    and never 0. Reporting 0 there would be the very defect this file is about,
    one field over.

GUARDS FAIL ON THE PRE-FIX TREE BY BEHAVIOUR, NOT BY A MISSING SYMBOL
---------------------------------------------------------------------
``probe(..., kind="llamacpp")`` does not raise on HEAD: every unknown kind is
caught and reported as ``reachable=False`` with ``unknown lane kind``. So the
guards below execute unchanged on both trees and fail on HEAD because the lane is
reported down, not because a name is absent.

Controls: the two existing kinds keep byte-identical behaviour, a llama.cpp box
still in its loading window is up-but-not-serving rather than a fake answer, and
the summary counts a llamacpp lane in the local denominator.
"""
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402


class _LlamaCpp(BaseHTTPRequestHandler):
    """A llama.cpp-shaped server: /health is keyless, /v1/models is key-gated."""

    #: 200 when a model is loaded; 503 during the load window.
    health_status = 200
    seen = []

    def do_GET(self):  # noqa: N802 - http.server API
        type(self).seen.append(self.path)
        if self.path == "/health":
            code = type(self).health_status
            body = json.dumps(
                {"status": "ok"} if code == 200
                else {"error": {"message": "Loading model", "code": 503}})
        elif self.path == "/v1/models":
            code = 401
            body = json.dumps({"error": {"message": "Invalid API Key",
                                         "type": "authentication_error",
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


def _serve(handler_cls, **attrs):
    handler = type("H", (handler_cls,), dict(seen=[], **attrs))
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", handler, srv


def _dead_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# --------------------------------------------------------------------------
# THE GAP
# --------------------------------------------------------------------------
def test_a_ready_llamacpp_lane_is_reported_serving_not_idle():
    """THE GUARD. /health 200 with a loaded model is capacity, not idleness.

    On HEAD this lane is reported reachable=False ("unknown lane kind"), which
    reads identically to a box that is off.
    """
    base, handler, srv = _serve(_LlamaCpp)
    try:
        row = fleet_probe.probe("q38", base, "llamacpp")
    finally:
        srv.shutdown()

    assert row["reachable"] is True, (
        "a llama.cpp server answering /health 200 was reported DOWN: %r"
        % (row["error"],))
    assert row["serving"] is True, (
        "a loaded llama.cpp model was not counted as serving, so this lane "
        "reads IDLE and the router falls through to a billed one")
    assert row["resident_count"] == 1


def test_a_key_gated_model_listing_is_unknown_not_zero():
    """A listing that FAILED is not an absence -- 401 must not become 0.

    Same rule the ollama path already follows for a failed /api/ps. Reporting 0
    here would restate the defect this file is about, one field over.
    """
    base, handler, srv = _serve(_LlamaCpp)
    try:
        row = fleet_probe.probe("q38", base, "llamacpp")
    finally:
        srv.shutdown()

    assert row["model_count"] is None, (
        "the model list answered 401 (key-gated) and the count became %r; a "
        "failed listing must be unknown, never empty"
        % (row["model_count"],))


def test_the_lane_is_in_the_default_fleet():
    """A lane kind nobody probes is the defect; it must be in LANES."""
    kinds = {kind for _label, _url, kind in fleet_probe.LANES}
    assert "llamacpp" in kinds, (
        "no llama.cpp lane is probed by default, so a serving llama.cpp server "
        "stays invisible on every run: %r" % (fleet_probe.LANES,))


def test_the_probe_reads_health_and_never_generates():
    """A probe must not perturb what it measures.

    /health is a state read. The generation routes are asserted absent so this
    lane cannot start a completion, which would evict the model it measures.
    """
    base, handler, srv = _serve(_LlamaCpp)
    try:
        fleet_probe.probe("q38", base, "llamacpp")
        seen = list(handler.seen)
    finally:
        srv.shutdown()

    assert "/health" in seen
    for forbidden in ("/completion", "/v1/chat/completions", "/v1/completions"):
        assert not any(forbidden in p for p in seen), forbidden


# --------------------------------------------------------------------------
# CONTROLS -- these must pass on HEAD as well, or the guards bound nothing
# --------------------------------------------------------------------------
def test_control_ollama_reading_is_unchanged():
    """The two existing kinds must behave exactly as before."""

    class _Ollama(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = ({"models": [{"name": "m1"}, {"name": "m2"}]}
                    if self.path == "/api/tags" else {"models": []})
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        row = fleet_probe.probe("3090", base, "ollama")
    finally:
        srv.shutdown()

    assert row["model_count"] == 2
    assert row["resident_count"] == 0
    assert row["serving"] is False, "an idle ollama lane must stay idle"


def test_a_loading_llamacpp_lane_is_up_but_not_serving():
    """503 is llama.cpp's load window: reachable, no capacity yet.

    A GUARD, not a control: it fails on HEAD because this lane kind does not
    exist there. Without it, "always report serving for llamacpp" would
    satisfy the first guard while lying during every model swap.
    """
    base, handler, srv = _serve(_LlamaCpp, health_status=503)
    try:
        row = fleet_probe.probe("q38", base, "llamacpp")
    finally:
        srv.shutdown()

    assert row["reachable"] is True, "a loading server is still reachable"
    assert row["serving"] is False, (
        "a server in its 503 load window was reported as serving capacity")


def test_control_a_dead_llamacpp_box_is_down_not_idle():
    """A box that is off must not be confused with one that is up and idle."""
    row = fleet_probe.probe("q38", f"http://127.0.0.1:{_dead_port()}",
                            "llamacpp", timeout=1.0)
    assert row["reachable"] is False
    assert row["error"], "a down lane must say why"


def test_a_serving_llamacpp_lane_enters_the_local_summary(monkeypatch):
    """The fleet line is the contract: it must count this lane and call it up.

    A GUARD: the stub row it builds is only reachable once this lane kind
    exists, so it cannot pass on HEAD.
    """
    import io
    from contextlib import redirect_stdout

    rows = [
        {"lane": "nous", "url": "http://x", "reachable": True,
         "models": ["m"], "model_count": 1, "resident": None,
         "resident_count": None, "serving": None, "resident_error": None,
         "latency_ms": 1, "error": None},
        {"lane": "q38", "url": "http://x", "reachable": True,
         "models": None, "model_count": None, "resident": ["<health:loaded>"],
         "resident_count": 1, "serving": True, "resident_error": None,
         "latency_ms": 1, "error": None},
    ]
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fleet_probe.main([])
    out = buf.getvalue()

    assert rc == 0
    assert "[1/1] local lanes SERVING" in out, (
        "a serving local lane must be counted in the denominator: %r" % (out,))
    assert "q38" in out
    # And the unknown model count must not crash the per-lane render.
    assert "installed" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
