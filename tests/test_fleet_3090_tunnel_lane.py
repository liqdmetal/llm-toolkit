"""The 3090 runs two servers; asking only its ollama daemon reported it IDLE.

THE FINDING, measured 2026-11-28 before this file existed
---------------------------------------------------------
``fleet_probe.py`` printed::

    3090  UP    7 installed, IDLE 0/7 loaded, 47 ms
    [1/4] local lanes SERVING (idle: 5090, 3090)

and a dispatch decision read from that would go to "the 3090 is free". It was
not. Two independent measurements of the same box at the same moment:

    $ curl -s http://192.168.1.124:11434/api/ps
    {"models":[]}                                   <- the ollama daemon

    $ curl -s -m 180 http://192.168.1.124:11434/api/generate \\
        -d '{"model":"defiant9b-abliterated:256k","prompt":"Say OK.",
             "stream":false,"options":{"num_predict":3}}'
    HTTP 500, in 4.0 s:
      llama-server process has terminated: exit status 1:
      ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed (out of
      memory), returning 0/0
      cudaMalloc failed: out of memory                      <- cannot load

    $ curl -s http://127.0.0.1:18082/health
    {"status":"ok"}                                 <- IS serving

The last one is the 3090's llama.cpp serving Inkling-2-Mini through an SSH
tunnel (``ssh -L 18082:127.0.0.1:8082 a3090@192.168.1.124``, confirmed by the
listening pid's own command line). So: the box was serving, its ollama daemon
reported nothing loaded and could not load anything, and no lane row asked the
server that answered. That is precisely the defect ``test_fleet_llamacpp_lane``
fixed for the 5090 in ``9b921d5``, one box over -- the fix added a llama.cpp
KIND but only ever pointed it at one machine.

WHY THE DIRECTION IS EXPENSIVE
------------------------------
This tool's stated reason to exist: "a reachable-but-idle lane is the expensive
default this tool exists to expose, because the router falls through to a
billed one." A box that is SERVING reported as idle produces two harms at once:
work that could run locally goes to a billed lane, AND a shaping dispatch onto
the "free" 3090 dies OOM on the card Inkling already owns.

The second harm has a second half worth naming: a tunnel answers on LOOPBACK.
Attributing it by URL host would have counted the 3090 as TWO machines -- one
serving at 127.0.0.1 and one idle-with-7-installed at 192.168.1.124 -- and the
idle half is the one a router reads.

WHAT IS ASSERTED
----------------
The lane must exist in LANES, a loading tunnel must be up-but-not-serving, and
a tunnel must be attributed to the machine it measures rather than to loopback.
The guards drive ``probe()`` and ``_host_of()`` rather than reading the LANES
tuple alone, because a lane that is listed but probes as down would satisfy a
membership assert while leaving the box still invisible.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

TUNNEL = "http://127.0.0.1:18082"
FAR_END = "192.168.1.124"


class _LlamaCpp(BaseHTTPRequestHandler):
    """A llama.cpp server: /health ready, model names behind a key gate."""

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            payload, status = json.dumps({"status": "ok"}).encode(), 200
        else:
            payload, status = b'{"error":"unauthorized"}', 401
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def _serve(handler, health_status=None):
    if health_status is not None:
        class _Loading(_LlamaCpp):
            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    payload = b'{"status":"loading model"}'
                    self.send_response(health_status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                super().do_GET()
        handler = _Loading
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


# ------------------------------------------------------------------ guards ---

def test_the_3090_serving_lane_is_in_the_default_fleet():
    """THE GUARD. A server nobody asks leaves a serving box looking idle."""
    urls = {url for _label, url, _kind in fleet_probe.LANES}
    assert TUNNEL in urls, (
        "no lane probes the 3090's llama.cpp server, so a box that is serving "
        "Inkling reads as IDLE on every run and work falls through to a billed "
        "lane: %r" % (fleet_probe.LANES,))


def test_the_tunnel_lane_is_probed_by_behaviour_not_just_listed():
    """Listed is not enough: the row must come back SERVING.

    The live tunnel is not available to a unit test, so this drives probe()
    against a stand-in that answers exactly what the real one answers --
    /health 200 and /v1/models 401.
    """
    base, srv = _serve(_LlamaCpp)
    try:
        row = fleet_probe.probe("3090-inkling", base, "llamacpp")
    finally:
        srv.shutdown()

    assert row["reachable"] is True, row["error"]
    assert row["serving"] is True, (
        "a llama.cpp server answering /health 200 was not counted as serving, "
        "so the 3090 reads IDLE while Inkling holds its card")
    assert row["model_count"] is None, (
        "a key-gated model listing became %r; a failed listing is unknown, "
        "never zero" % (row["model_count"],))


def test_a_tunnel_is_attributed_to_the_machine_it_measures():
    """A tunnel answers on loopback; the GPU is at the far end."""
    assert fleet_probe._host_of(TUNNEL) == FAR_END, (
        "the 3090's tunnel was attributed to %r, so the same card is counted "
        "once as serving on loopback and once as idle at its real address -- "
        "and the idle half is what a router reads"
        % (fleet_probe._host_of(TUNNEL),))


def test_the_3090_is_one_host_not_two():
    """The consequence of the attribution: one GPU, one host entry."""
    rows = [
        {"lane": "3090", "url": "http://192.168.1.124:11434",
         "reachable": True, "models": ["m"], "model_count": 7,
         "resident": [], "resident_count": 0, "serving": False,
         "resident_error": None, "latency_ms": 1, "error": None},
        {"lane": "3090-inkling", "url": TUNNEL,
         "reachable": True, "models": None, "model_count": None,
         "resident": ["<loaded; name key-gated>"], "resident_count": 1,
         "serving": True, "resident_error": None, "latency_ms": 1,
         "error": None},
    ]
    by_host, free, shared, unread = fleet_probe.host_capacity(rows)

    assert list(by_host) == [FAR_END], (
        "the 3090 split into %d hosts (%r); a tunnel must not create a second "
        "machine" % (len(by_host), list(by_host)))
    assert free == [], (
        "a host with a serving daemon was counted as free capacity: %r" % (free,))
    assert "3090" in shared, (
        "the idle ollama daemon was not named as an idle-daemon-on-a-busy-host, "
        "so its idleness still reads as spare capacity: %r" % (shared,))


# ---------------------------------------------------------------- controls ---

def test_control_a_loading_tunnel_is_up_but_not_serving():
    """503 is llama.cpp's load window. Passes on HEAD's kind too, and stops
    "always say serving" from satisfying the guard above."""
    base, srv = _serve(_LlamaCpp, health_status=503)
    try:
        row = fleet_probe.probe("3090-inkling", base, "llamacpp")
    finally:
        srv.shutdown()

    assert row["reachable"] is True, "a loading server is still reachable"
    assert row["serving"] is False, (
        "a server in its 503 load window was reported as serving capacity")


def test_control_a_real_far_end_hostname_still_resolves_to_itself():
    """The tunnel map must not swallow ordinary lanes."""
    for url, want in (
        ("http://192.168.1.232:11434", "192.168.1.232"),
        ("http://192.168.1.232:8081", "192.168.1.232"),
        ("http://127.0.0.1:8099", "127.0.0.1"),
    ):
        assert fleet_probe._host_of(url) == want, url


def test_control_two_tunnels_to_one_box_are_still_one_host():
    """A second forwarded port must not become a third machine."""
    assert fleet_probe._host_of(TUNNEL) == fleet_probe._host_of(
        "http://192.168.1.124:11434")


# ---------------------------------------------------------------------------

def test_the_probe_still_never_generates():
    """The standing rule: a probe must not evict what it measures.

    /health is a state read; the generation routes must never be touched.
    """
    seen = []

    class _Recorder(_LlamaCpp):
        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            super().do_GET()

    base, srv = _serve(_Recorder)
    try:
        fleet_probe.probe("3090-inkling", base, "llamacpp")
    finally:
        srv.shutdown()

    assert "/health" in seen
    for forbidden in ("/completion", "/v1/chat/completions", "/v1/completions",
                      "/api/generate"):
        assert not any(forbidden in p for p in seen), forbidden


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
