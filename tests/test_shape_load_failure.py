"""A model that is LISTED but cannot be LOADED must SKIP, not break the caller.

ROUTING.md rule 1: shaping is a **soft** dependency -- "shaper failure falls
through to the raw task". A caller is told to catch ``ShaperUnavailable``.

``preflight()`` answers "is the model installed" by reading ``/api/tags``. That
question and "can the box actually serve it" are different questions, and the
shaper box is shared with a resident model that can hold the whole card. So a
model can be present in the listing and still be unloadable at generation time.

MEASURED 2026-11-28 against the real 3090 (192.168.1.124:11434), with
``defiant9b-abliterated:256k`` -- ``shape.MODEL``'s own default -- listed and
Inkling holding the VRAM:

    /api/ps  -> {"models": []}                    (nothing resident of ours)
    POST /api/generate -> HTTP 500, body:
        llama-server process has terminated: exit status 1:
        ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed (out of
        memory), returning 0/0 ... cudaMalloc failed: out of memory ...
        failed to allocate CUDA0 buffer of size 52690944 ...
        llama_init_from_model: failed to initialize the context

At HEAD 754788e the CLI printed the generic-failure marker for that:

    SHAPE_FAIL: HTTP Error 500: Internal Server Error      exit 1

``main()`` uses SHAPE_FAIL for "python raised something", and SHAPE_SKIP for
ShaperUnavailable. A caller following ROUTING.md rule 1 keys off that, so the
one failure the shared box produces most often -- the card is full -- was
reported as the one thing the caller is not told to handle.

What this file pins
-------------------
1. A 500 whose body says the model could not be brought up raises
   ``ShaperUnavailable``: the caller skips to its own prompt.
2. The message says WHICH model and WHY, or the operator cannot tell "your
   config names a model that is not there" from "the card is busy".
3. NEGATIVE CONTROL, and it is the half that keeps the fix honest: an unrelated
   HTTP 500 -- a server bug, not a capacity problem -- must still RAISE. Without
   it, "convert 500 to skip" silently converts every server fault into capacity
   loss that nobody sees.
4. A 200 is untouched.
"""
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import shape  # noqa: E402

INSTALLED = "defiant9b-abliterated:256k"

# The real body, verbatim from the run recorded in the module docstring
# (elision marked, not a paraphrase of the mechanism).
OOM_BODY = (
    "llama-server process has terminated: exit status 1: "
    "ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed (out of "
    "memory), returning 0/0\ncudaMalloc failed: out of memory\n"
    "ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA0 "
    "buffer of size 52690944\nllama_init_from_model: failed to initialize "
    "the context: failed to allocate buffer for rs cache"
)

# Not a capacity problem: something in the server broke. Must stay fatal.
SERVER_BUG_BODY = '{"error": "internal: nil map dereference at runtime.go:412"}'


class _Stub:
    """The two endpoints shape.py touches, with a controllable generate reply."""

    def __init__(self, models, code=200, body=None, reply="shaped task"):
        self.models = list(models)
        self.code = code
        self.body = body
        self.reply = reply
        self.generated = []

    def start(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                payload = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/api/tags":
                    return self._send(
                        200, {"models": [{"name": n} for n in stub.models]})
                if self.path == "/api/ps":
                    return self._send(200, {"models": []})
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                stub.generated.append(payload.get("model"))
                if stub.code != 200:
                    return self._send(stub.code, {"error": stub.body})
                return self._send(200, {"response": stub.reply})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def _wire(monkeypatch, stub):
    srv = stub.start()
    monkeypatch.setattr(shape, "OLLAMA",
                        "http://127.0.0.1:%d" % srv.server_address[1])
    monkeypatch.setattr(shape, "MODEL", INSTALLED)
    monkeypatch.setattr(shape, "PROBE_TIMEOUT", 2.0)
    monkeypatch.delenv("SHAPE_DISABLE", raising=False)
    return stub, srv


def test_a_listed_model_that_cannot_load_raises_ShaperUnavailable(monkeypatch):
    """THE RED TEST. The card being full must take the documented skip path."""
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED], code=500, body=OOM_BODY))
    try:
        with pytest.raises(shape.ShaperUnavailable) as ei:
            shape.shape("summarise this log")
        msg = str(ei.value)
        # The operator has to be able to tell this apart from a config typo.
        assert INSTALLED in msg, msg
        assert "could not be loaded" in msg, msg
    finally:
        srv.shutdown()


def test_the_skip_is_reachable_through_the_cli_marker(monkeypatch):
    """The distinction that reaches an operator is a marker, not an exception.

    ROUTING.md rule 1 is written for a caller invoking this as a subprocess
    (`exit 1` -> caller uses original), so the fix has to survive the exit path,
    not just the library call. That means running the real `__main__` block
    against the stub rather than calling `shape()` and looking for SystemExit --
    the marker is produced by `__main__`, which importing the module skips.
    """
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED], code=500, body=OOM_BODY))
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(shape.__file__))
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "shape.py"), "summarise this log"],
            cwd=here,
            env={**os.environ,
                 "SHAPE_OLLAMA": shape.OLLAMA,
                 "SHAPE_MODEL": INSTALLED,
                 "SHAPE_TIMEOUT": "20"},
            capture_output=True, timeout=90)
        assert proc.returncode == 1, proc.stderr
        err = proc.stderr.decode("utf-8", "replace")
        # The operator-facing half: SKIP, not the generic failure marker.
        assert err.startswith("SHAPE_SKIP:"), err
        assert "SHAPE_FAIL" not in err, err
    finally:
        srv.shutdown()


def test_control_an_unrelated_500_still_raises(monkeypatch):
    """NEGATIVE CONTROL: a server bug must not be relabelled a capacity skip.

    This is the assertion that fails under the lazy version of the fix --
    `except HTTPError: raise ShaperUnavailable` -- which would report every
    backend fault as a quiet skip and lose capacity with no signal.
    """
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED], code=500,
                                         body=SERVER_BUG_BODY))
    try:
        with pytest.raises(Exception) as ei:
            shape.shape("summarise this log")
        assert not isinstance(ei.value, shape.ShaperUnavailable), (
            "an unrelated server fault was reported as a shaper skip (%r)"
            % (ei.value,))
    finally:
        srv.shutdown()


def test_control_a_healthy_shaper_still_shapes(monkeypatch):
    """CONTROL: the happy path is untouched by the new error branch."""
    stub, srv = _wire(monkeypatch, _Stub([INSTALLED], reply="rewritten prompt"))
    try:
        assert shape.shape("summarise this log") == "rewritten prompt"
        assert stub.generated == [INSTALLED]
    finally:
        srv.shutdown()


def test_the_marker_test_is_not_vacuous_on_empty_bodies():
    """MUTATION BATTERY: `_is_model_load_failure` must not answer True always.

    A predicate that returned True for everything would satisfy the red test and
    quietly convert every 500 into a skip. These are the cases that pin it.
    """
    assert shape._is_model_load_failure(OOM_BODY) is True
    assert shape._is_model_load_failure(SERVER_BUG_BODY) is False
    assert shape._is_model_load_failure("") is False
    assert shape._is_model_load_failure("rate limit exceeded") is False
