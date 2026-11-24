"""A shaper with the wrong model must SKIP, not break the caller.

ROUTING.md rule 1: "`.shape.py` is a **soft** dependency: shaper failure falls
through to the raw task (`exit 1` -> caller uses original). Frontline work must
never hard-block on the 3090." A caller is therefore told to catch
``ShaperUnavailable`` and use its own prompt.

MEASURED at HEAD 106bd60, against the real 3090 configured in ``shape.OLLAMA``:

    SHAPE_MODEL=<installed>        shape() -> shaped prompt
    SHAPE_MODEL=<not installed>    shape() -> HTTPError 404 ESCAPES
    host down                      shape() -> ShaperUnavailable (correct)

The middle row is the defect. ``preflight()`` only proves the PORT is open, so a
model name that does not exist on the shaper host sails through the guard and
dies at the HTTP layer as a generic ``urllib.error.HTTPError`` -- an exception
type no caller is told to catch. The documented soft-skip path therefore does not
catch it, and "shaping must never block the caller" is broken by the one failure
a config typo produces. Measured end-to-end through the CLI: ``SHAPE_FAIL: HTTP
Error 404: Not Found`` where the contract calls for ``SHAPE_SKIP: ...``.

Measured cost of the wrong answer: the 404 returns in 11 ms. This is not a
timeout problem, it is a contract problem -- the caller gets an exception type it
was told cannot happen.

Two things this file pins, kept separate on purpose
---------------------------------------------------
1. ``preflight()`` must ask whether the MODEL is installed, not only whether the
   port is open. ``/api/tags`` is a listing endpoint: it reads state and loads
   nothing, so asking is free and cannot evict a resident model.
2. A 404 must ALSO be converted to ``ShaperUnavailable`` when it happens. The
   installed-model set can change between the preflight and the generation, and
   the contract is about what the CALLER sees, not about how early we notice.

The control half matters as much: a shaper whose configured model IS present must
still shape, and a genuinely unreachable host must still skip for its ORIGINAL
reason. A "fix" that turned every shaper call into a skip would satisfy half of
these assertions and must not satisfy the rest.
"""
import json
import os
import socket
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import shape  # noqa: E402

INSTALLED = "defiant9b-abliterated:256k"
NOT_INSTALLED = "definitely-not-installed-xyz:1b"


class _Stub:
    """A minimal fake of the three Ollama endpoints shape.py touches.

    Serving /api/tags, /api/ps and /api/generate from one small local HTTP server
    keeps these tests off the 3090 entirely: that box is Hermes-reserved, and a
    test needing it live would be a test nobody could run. It also makes each
    failure mode explicit instead of depending on what the box happens to have
    installed today.

    ``broken_tags`` models the case that must NOT be read as an absence: a
    listing endpoint that errors. A missing model and an unreadable model list
    are different facts, and a guard that confuses them refuses every model on a
    box whose /api/tags is merely unhealthy.
    """

    def __init__(self, models, reply="shaped task", broken_tags=False):
        self.models = list(models)
        self.reply = reply
        self.broken_tags = broken_tags
        self.generated = []      # every model actually asked to generate
        self.paths = []

    def start(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                stub.paths.append(self.path)
                if self.path == "/api/tags":
                    if stub.broken_tags:
                        return self._json(500, {"error": "listing unavailable"})
                    return self._json(200,
                                      {"models": [{"name": n} for n in stub.models]})
                if self.path == "/api/ps":
                    return self._json(200, {"models": []})
                return self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                model = payload.get("model")
                stub.generated.append(model)
                if self.path != "/api/generate":
                    return self._json(404, {"error": "not found"})
                if model not in stub.models:
                    # Verbatim the shape the real box answered, measured:
                    # HTTP 404 {"error": "model '...' not found"}
                    return self._json(404, {"error": "model %r not found" % model})
                return self._json(200, {"response": stub.reply})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def _wire(monkeypatch, stub, model):
    """Start a stub and point shape.py at it. Returns (stub, server)."""
    srv = stub.start()
    monkeypatch.setattr(shape, "OLLAMA",
                        "http://127.0.0.1:%d" % srv.server_address[1])
    monkeypatch.setattr(shape, "MODEL", model)
    monkeypatch.setattr(shape, "PROBE_TIMEOUT", 2.0)
    monkeypatch.delenv("SHAPE_DISABLE", raising=False)
    return stub, srv


def test_a_configured_model_that_is_installed_still_shapes(monkeypatch):
    """The control that must keep passing: the happy path is untouched."""
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED], reply="rewritten prompt"),
                      INSTALLED)
    try:
        assert shape.shape("summarise this log") == "rewritten prompt"
        assert stub.generated == [INSTALLED]
    finally:
        srv.shutdown()


def test_a_missing_model_raises_ShaperUnavailable_not_HTTPError(monkeypatch):
    """THE RED TEST. A config typo must take the documented soft-skip path."""
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED]), NOT_INSTALLED)
    try:
        with pytest.raises(shape.ShaperUnavailable) as ei:
            shape.shape("summarise this log")
        # The caller needs to know WHICH model, or a config typo stays
        # undiagnosable from the message.
        assert NOT_INSTALLED in str(ei.value), str(ei.value)
    finally:
        srv.shutdown()


def test_preflight_refuses_a_model_the_host_does_not_list(monkeypatch):
    """Refusing EARLY is the second half: answer before the generation request."""
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED]), NOT_INSTALLED)
    try:
        with pytest.raises(shape.ShaperUnavailable) as ei:
            shape.preflight()
        assert NOT_INSTALLED in str(ei.value), str(ei.value)
        assert stub.generated == [], (
            "preflight started a generation (%r) while checking for a model; a "
            "probe that loads a model evicts the one it is measuring"
            % (stub.generated,))
    finally:
        srv.shutdown()


def test_preflight_passes_when_the_model_is_listed(monkeypatch):
    """The control: the widened guard must not refuse a correct config."""
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED]), INSTALLED)
    try:
        shape.preflight()          # must not raise
        assert stub.generated == []
    finally:
        srv.shutdown()


def test_a_host_that_cannot_be_reached_still_skips_for_the_original_reason(
        monkeypatch):
    """A dead box and a missing model must not become the same message.

    Both take the soft-skip path, so the exception TYPE matches; the REASON must
    still separate "the box is not there" from "the model is not there".
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    monkeypatch.setattr(shape, "OLLAMA", "http://127.0.0.1:%d" % port)
    monkeypatch.setattr(shape, "MODEL", INSTALLED)
    monkeypatch.setattr(shape, "PROBE_TIMEOUT", 1.0)
    monkeypatch.delenv("SHAPE_DISABLE", raising=False)
    with pytest.raises(shape.ShaperUnavailable) as ei:
        shape.preflight()
    assert "unreachable" in str(ei.value).lower(), str(ei.value)
    assert INSTALLED not in str(ei.value), (
        "an unreachable box blamed the model: %r" % (str(ei.value),))


def test_control_a_broken_model_listing_is_not_an_absent_model(monkeypatch):
    """A listing that ERRORS must not manufacture an absence.

    If the widened guard treated an unreadable /api/tags as "no models", a box
    whose listing endpoint was unhealthy would refuse every model -- turning a
    soft dependency into a guaranteed skip. Only a listing that ANSWERS may be
    read as authoritative.
    """
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED], reply="anyway",
                                        broken_tags=True), INSTALLED)
    try:
        assert shape.shape("summarise this log") == "anyway"
    finally:
        srv.shutdown()


def test_control_a_404_at_generation_time_also_skips(monkeypatch):
    """The model can vanish between preflight and generation.

    Pinned separately so the fix cannot be preflight-only: the contract is about
    what the caller sees, and a shaper that passes the model check then 404s must
    still skip rather than raise HTTPError.
    """
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED]), INSTALLED)
    try:
        # Make the preflight succeed, then have the model disappear.
        shape.preflight()
        stub.models = []
        with pytest.raises(shape.ShaperUnavailable) as ei:
            shape.shape("summarise this log")
        assert INSTALLED in str(ei.value), str(ei.value)
    finally:
        srv.shutdown()
