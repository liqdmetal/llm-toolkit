"""Pins for fleet_probe.py — lane state must be measured, never assumed.

Why this exists: ROUTING.md routes work by lane, and until now nothing in this
repo could answer "is that lane up, and does it carry the model the policy
names?". Both failures look identical from the router's side — a request goes
out and nothing comes back — and both end at the same expensive default: a
billed frontier lane. The probe turns that into a number per lane.

The guard that matters here is not "the probe returns a dict". It is that a DOWN
lane and an UP lane are distinguishable AT ALL, and that the probe never loads a
model. A probe that started a generation would evict a resident model and change
the thing it was sent to measure, so test_listing_endpoints_only pins the URL
shapes.

Live measurement, 2026-11-24 (this is the finding, not a fixture):

    nous UP   410 models, 316 ms
    5090 UP     3 models  (7b-qwen-q4, qwen2.5:7b, llama3.2-vision)
    3090 UP     7 models  (incl. defiant9b-abliterated:256k)

ROUTING.md rule 1 sends sensitive reasoning to "q38 jumbo, defiant abliterated,
Qwen heretic on the 5090". Two of those three are absent from the box it names.
That row cannot execute as written — which is exactly the drift this file exists
to catch.

    python -m pytest tests/test_fleet_probe.py -q
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


class _Handler(BaseHTTPRequestHandler):
    """Serves one canned JSON body for any GET; records the paths requested."""

    body = b"{}"
    seen = []

    def do_GET(self):  # noqa: N802 - http.server API
        type(self).seen.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):  # keep pytest output clean
        pass


def _serve(body):
    """Start a local server returning `body`; yields (base_url, handler class)."""
    handler = type("H", (_Handler,), {"body": json.dumps(body).encode(), "seen": []})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{srv.server_address[1]}", handler, srv


def _dead_port():
    """A port nothing listens on, obtained by binding then releasing."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def test_probe_reports_an_up_lane_with_its_models():
    base, handler, srv = _serve({"models": [{"name": "defiant9b-abliterated:256k"},
                                            {"name": "qwen3.5:27b"}]})
    try:
        row = fleet_probe.probe("3090", base, "ollama")
    finally:
        srv.shutdown()
    assert row["reachable"] is True
    assert row["model_count"] == 2
    assert "defiant9b-abliterated:256k" in row["models"]
    assert row["error"] is None


def test_probe_reports_a_down_lane_without_raising():
    """A down lane is DATA, not a crash: the whole point is to report it."""
    row = fleet_probe.probe("dead", f"http://127.0.0.1:{_dead_port()}", "ollama",
                            timeout=1.0)
    assert row["reachable"] is False
    assert row["model_count"] == 0
    assert row["error"], "a down lane must say why, not just report zero"


def test_control_up_and_down_are_distinguishable():
    """The load-bearing assertion: an all-down probe and an all-up probe differ.

    Without this, a probe that reported reachable=False for everything would
    satisfy the down-lane test and every row would look like an outage.
    """
    base, handler, srv = _serve({"models": [{"name": "m"}]})
    try:
        up = fleet_probe.probe("up", base, "ollama")
    finally:
        srv.shutdown()
    down = fleet_probe.probe("down", f"http://127.0.0.1:{_dead_port()}", "ollama",
                             timeout=1.0)
    assert up["reachable"] != down["reachable"]


def test_openai_lane_reads_the_dot_v1_models_path():
    base, handler, srv = _serve({"data": [{"id": "z-ai/glm-5.3-flashx"}]})
    try:
        row = fleet_probe.probe("nous", base, "openai")
    finally:
        srv.shutdown()
    assert row["models"] == ["z-ai/glm-5.3-flashx"]


def test_listing_endpoints_only_never_start_a_generation():
    """A probe must not perturb what it measures.

    Loading a model to answer a probe would evict a resident one and change the
    lane's state — the probe would then be measuring itself. Both listing paths
    are asserted to be the cheap ones, and the paths that would generate are
    rejected.
    """
    base, handler, srv = _serve({"models": [], "data": []})
    try:
        fleet_probe.probe("o", base, "ollama")
        fleet_probe.probe("v", base, "openai")
        seen = list(handler.seen)
    finally:
        srv.shutdown()
    assert seen == ["/api/tags", "/v1/models"], seen
    for forbidden in ("/api/generate", "/api/chat", "/v1/chat/completions"):
        assert not any(forbidden in p for p in seen), forbidden


def test_unknown_lane_kind_is_a_down_lane_not_a_crash():
    row = fleet_probe.probe("x", "http://127.0.0.1:1", "bogus")
    assert row["reachable"] is False
    assert "unknown lane kind" in row["error"]


def test_malformed_body_is_a_down_lane_not_a_crash():
    """A lane answering non-JSON must not take the probe down with it."""
    base, handler, srv = _serve({})
    handler.body = b"<html>cloudflare interstitial</html>"
    try:
        row = fleet_probe.probe("bad", base, "ollama")
    finally:
        srv.shutdown()
    assert row["reachable"] is False
    assert row["error"]


def test_probe_all_returns_one_row_per_lane():
    rows = fleet_probe.probe_all(
        [("a", f"http://127.0.0.1:{_dead_port()}", "ollama"),
         ("b", f"http://127.0.0.1:{_dead_port()}", "openai")], timeout=1.0)
    assert [r["lane"] for r in rows] == ["a", "b"]
