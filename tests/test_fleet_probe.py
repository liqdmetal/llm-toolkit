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
    lane's state — the probe would then be measuring itself. Every listing path
    is asserted to be the cheap ones, and the paths that would generate are
    rejected.

    /api/ps is on this list deliberately: it is the residency read. It reports
    which models are already loaded and loads none, which is what makes
    "installed" and "serving" separable without perturbing either.
    """
    base, handler, srv = _serve({"models": [], "data": []})
    try:
        fleet_probe.probe("o", base, "ollama")
        fleet_probe.probe("v", base, "openai")
        seen = list(handler.seen)
    finally:
        srv.shutdown()
    assert seen == ["/api/tags", "/api/ps", "/v1/models"], seen
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


def test_probe_separates_installed_from_serving():
    """Seven models installed and none loaded is UP and IDLE, not "7 models".

    This is the defect the residency read exists for: the probe printed a lane
    with nothing resident identically to one actively serving, so an idle GPU
    read as healthy capacity while work went to a billed lane.
    """
    base, handler, srv = _serve({"models": [{"name": "a"}, {"name": "b"}]})
    try:
        row = fleet_probe.probe("3090", base, "ollama")
    finally:
        srv.shutdown()
    assert row["model_count"] == 2
    # The canned body is the same dict for both paths, so /api/ps "reports"
    # two loaded here. The point of the test is that the two counts are
    # carried SEPARATELY; a fixed body cannot make them agree by accident.
    assert row["resident"] is not None
    assert row["resident_count"] == 2


def test_control_an_idle_lane_is_not_a_serving_lane():
    """The load-bearing one: zero resident must not read as zero installed."""
    class Handler(_Handler):
        def do_GET(self):  # noqa: N802
            Handler.seen.append(self.path)
            body = ({"models": [{"name": "m1"}, {"name": "m2"}, {"name": "m3"}]}
                    if self.path == "/api/tags" else {"models": []})
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    handler = type("Idle", (Handler,), {"seen": []})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        row = fleet_probe.probe("3090", base, "ollama")
    finally:
        srv.shutdown()

    assert row["reachable"] is True, 'the process answers, so it is UP'
    assert row["model_count"] == 3, 'three models are installed'
    assert row["resident_count"] == 0, 'but none are loaded'
    assert row["serving"] is False, 'and that is IDLE, which is the finding'
    assert row["model_count"] != row["resident_count"],         'if these could not differ the distinction would be vacuous'


def test_control_a_hosted_lane_reports_residency_as_not_applicable():
    """None ("does not apply") must not be confused with [] ("nothing loaded").

    A hosted portal has no resident model set, so reporting [] would claim it
    was idle — a false idle on the one lane that is always able to answer.
    """
    base, handler, srv = _serve({"data": [{"id": "z-ai/glm-5.3-flashx"}]})
    try:
        row = fleet_probe.probe("nous", base, "openai")
    finally:
        srv.shutdown()
    assert row["resident"] is None
    assert row["resident_count"] is None
    assert row["serving"] is None, 'not False: "unknown/NA" is not "idle"'


def test_residency_failure_is_reported_and_does_not_downgrade_the_lane():
    """A lane that lists models but cannot answer /api/ps is still UP."""
    class Handler(_Handler):
        def do_GET(self):  # noqa: N802
            Handler.seen.append(self.path)
            if self.path == "/api/ps":
                self.send_response(500)
                self.end_headers()
                return
            payload = json.dumps({"models": [{"name": "m"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    handler = type("H2", (Handler,), {"seen": []})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        row = fleet_probe.probe("x", base, "ollama")
    finally:
        srv.shutdown()

    assert row["reachable"] is True
    assert row["model_count"] == 1
    assert row["resident"] is None
    assert row["serving"] is None
    assert row["resident_error"], 'an unanswered residency question says so'
