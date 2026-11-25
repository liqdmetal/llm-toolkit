"""A key-gated local server's spare capacity must be readable, not eternally UNKNOWN.

WHAT WAS MEASURED BEFORE THIS FILE EXISTED (2026-11-29, all three local
llama.cpp lanes read keyless from the box):

    q38           /health 200   /v1/models 401   /slots 401
    3090-inkling  /health 200   /v1/models 401   /slots 401
    3090-defiant  /health 200   /v1/models 200   /slots 200

So /health is the only keyless read, and it answers exactly one bit -- a model
is loaded. Spare capacity lives in /slots, and TWO of THREE loaded servers refuse
it without a bearer token. ``fleet_probe`` had no way to send one, so those lanes
printed ``slots UNKNOWN`` on every run, forever. Unknown is not neutral: the
router reads it as "do not dispatch there" and falls through to the billed lane
this tool exists to keep it off. Capacity that demonstrably exists and is never
measured is the same expensive default as capacity that is missing.

THE DEFECT IS THE MISSING PARAMETER, NOT THE 401. A 401 is a correct answer and
must keep rendering as UNKNOWN -- the ``test_fleet_slots_are_not_residency``
suite pins that and this file does not disturb it. What changes is that a
configured key now turns the same endpoint into a measurement instead of a
permanent shrug.

THE SECRET IS NEVER IN THE ROW. The row records WHICH variable supplied the key
(``key_source``); the value goes only into the Authorization header. A report
that echoes a key to a terminal and into ``--json`` is a leak that outlives the
run, so one arm below asserts the value appears in no field of the output at all.

GUARDS FAIL ON THE PRE-FIX TREE BY VALUE, NOT BY A MISSING SYMBOL. Every guard
drives ``probe_all()``/``main()`` -- the public path -- and reads the row or the
rendered line. Pre-fix, the env var is simply ignored: the keyed server answers
401 and the row shows ``slots=None``/``models=None``.

This file therefore spells the variable names as LITERALS and never as
``fleet_probe.LANE_KEY_ENV``. The first draft did the latter and the negative
control came back 11/11 red on `AttributeError: module 'fleet_probe' has no
attribute 'LANE_KEY_ENV'` -- every arm, controls included, i.e. measuring the
absence of a constant instead of the defect. A guard that fails on a missing
symbol bounds nothing, and a control that cannot pass on the pre-fix tree is not
a control. The literals below are the contract with the operator anyway: the
names are what a human types into a shell.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

# The operator-facing contract, spelled out rather than imported (see above).
SHARED_KEY_VAR = "FLEET_LANE_API_KEY"
PER_LANE_PREFIX = "FLEET_LANE_API_KEY_"


def per_lane_var(label):
    """The same rule the tool documents: upper-case, non-alphanumerics -> _."""
    return PER_LANE_PREFIX + "".join(
        c if c.isalnum() else "_" for c in label).upper()


# ---------------------------------------------------------------------------
# A llama.cpp server that key-gates /v1/models and /slots and leaves /health
# open -- the shape MEASURED on q38 and the Inkling tunnel, not an invention.
# Configuration is mutated on the class between tests, the pattern the existing
# fleet suites use, so the handler keeps http.server's own __init__.
# ---------------------------------------------------------------------------
class _KeyedLlama(BaseHTTPRequestHandler):
    require_key = "test-secret-value"
    slots = [{"is_processing": False}] * 4
    models = [{"id": "some-model.gguf"}, {"id": "other.gguf"}]
    seen_auth = []

    def log_message(self, *a):  # keep pytest output clean
        pass

    def _authorised(self):
        cls = type(self)
        header = self.headers.get("Authorization", "")
        cls.seen_auth.append(header)
        if not cls.require_key:
            return True
        return header == "Bearer " + cls.require_key

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        cls = type(self)
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        if not self._authorised():
            self._send(401, {"error": {"message": "Invalid API Key",
                                       "type": "authentication_error"}})
            return
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": cls.models})
            return
        if self.path == "/slots":
            self._send(200, cls.slots)
            return
        self._send(404, {"error": {"message": "not found"}})


@pytest.fixture()
def keyed():
    """A live keyed server; yields its base URL and the class for inspection."""
    _KeyedLlama.require_key = "test-secret-value"
    _KeyedLlama.slots = [{"is_processing": False}] * 4
    _KeyedLlama.models = [{"id": "some-model.gguf"}, {"id": "other.gguf"}]
    _KeyedLlama.seen_auth = []
    srv = HTTPServer(("127.0.0.1", 0), _KeyedLlama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _clear_keys(monkeypatch):
    """No lane key configured -- the default deployment on this box."""
    for name in list(os.environ):
        if name == SHARED_KEY_VAR or name.startswith(PER_LANE_PREFIX):
            monkeypatch.delenv(name, raising=False)


def _render(monkeypatch, rows):
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    return "\n".join(_capture(monkeypatch))


def _capture(monkeypatch):
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        fleet_probe.main([])
    return buf.getvalue().splitlines()


# ---------------------------------------------------------------------------
# THE GUARDS -- each fails by value on the pre-fix tree
# ---------------------------------------------------------------------------
def test_a_configured_key_turns_a_refused_slot_read_into_a_measurement(
        keyed, monkeypatch):
    """THE defect. Without the fix this lane is UNKNOWN on every run forever."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv(SHARED_KEY_VAR, "test-secret-value")
    row = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])[0]
    assert row["slots"] == (4, 0), (
        "a configured key did not reach /slots, so spare capacity on a "
        "key-gated server is unmeasurable: slots=%r slots_error=%r"
        % (row.get("slots"), row.get("slots_error")))


def test_a_configured_key_also_reads_the_model_list(keyed, monkeypatch):
    """The same gate covers /v1/models, so the installed count was fabricated
    as unknown too. Both endpoints must come back with one key."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv(SHARED_KEY_VAR, "test-secret-value")
    row = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])[0]
    assert row["model_count"] == 2, (
        "the key reached /slots but not /v1/models: model_count=%r"
        % (row.get("model_count"),))


def test_the_row_names_which_variable_supplied_the_key(keyed, monkeypatch):
    """A report must say which knob to turn, and never what it contains."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv(SHARED_KEY_VAR, "test-secret-value")
    row = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])[0]
    assert row.get("key_source") == SHARED_KEY_VAR, (
        "the row did not record which variable supplied the key: %r"
        % (row.get("key_source"),))


def test_a_per_lane_variable_beats_the_shared_fallback(keyed, monkeypatch):
    """Per-lane precedence, exercised through the live server: with both set and
    only the per-lane one correct, the read must still succeed."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv(SHARED_KEY_VAR, "wrong-shared-key")
    monkeypatch.setenv(per_lane_var("keyed"), "test-secret-value")
    row = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])[0]
    assert row["slots"] == (4, 0), (
        "the per-lane variable did not take precedence over the fallback: "
        "slots=%r slots_error=%r" % (row.get("slots"), row.get("slots_error")))
    assert row["key_source"] == per_lane_var("keyed")


def test_an_unreadable_lane_names_the_knob_that_would_fix_it(monkeypatch):
    """The render half: 'slots UNKNOWN' with no cause is what made this
    invisible. The operator must be told which variable to set."""
    _clear_keys(monkeypatch)
    rows = [{
        "lane": "gated", "url": "http://127.0.0.1:1", "reachable": True,
        "models": None, "model_count": None,
        "resident": ["<loaded; name key-gated>"], "resident_count": 1,
        "serving": True, "resident_error": None,
        "slots": None, "slots_error": "HTTPError: HTTP Error 401: Unauthorized",
        "key_source": None,
        "latency_ms": 1, "error": None,
    }]
    out = _render(monkeypatch, rows)
    assert "no API key configured" in out, out
    assert SHARED_KEY_VAR in out, out
    assert per_lane_var("gated") in out, out


# ---------------------------------------------------------------------------
# THE CONTROLS -- these must pass on BOTH trees, so a guard cannot be green
# for the wrong reason
# ---------------------------------------------------------------------------
def test_without_a_key_the_refusal_still_renders_as_unknown(keyed, monkeypatch):
    """ABSENT != ZERO survives this change. An unread lane is never 0 busy and
    never 'full', and must never be counted as free capacity."""
    _clear_keys(monkeypatch)
    row = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])[0]
    assert row["slots"] is None, row["slots"]
    assert row.get("slots_error"), "the refusal was not recorded at all"
    assert "401" in row["slots_error"], row["slots_error"]
    out = _render(monkeypatch, [row])
    assert "slots UNKNOWN" in out, out
    assert "0/4 slots busy" not in out, (
        "an unread lane was rendered as a measured empty server: %s" % out)


def test_a_keyless_server_is_unchanged(keyed, monkeypatch):
    """A server that needs no key must behave exactly as before, with or
    without one configured -- the fix must not require a key to work."""
    _clear_keys(monkeypatch)
    _KeyedLlama.require_key = ""
    row = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])[0]
    assert row["slots"] == (4, 0), row["slots"]
    assert row["model_count"] == 2, row["model_count"]
    assert row.get("key_source") is None, (
        "a lane that was read without a key claimed a key source: %r"
        % (row.get("key_source"),))

def test_the_secret_never_appears_in_the_row(keyed, monkeypatch):
    """A key that reaches a terminal or a --json file outlives the run. Only the
    variable NAME may be recorded."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv(SHARED_KEY_VAR, "test-secret-value")
    rows = fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])
    blob = json.dumps(rows)
    assert "test-secret-value" not in blob, (
        "the API key value was written into the report: %s" % blob)


def test_the_key_is_actually_sent_as_a_bearer_token(keyed, monkeypatch):
    """GUARD, not a control: with no key support the header is never sent, so
    this fails by value pre-fix. It asserts the header is well-formed rather
    than merely present, because a server shaped like this one rejects
    anything else."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv(SHARED_KEY_VAR, "test-secret-value")
    _KeyedLlama.seen_auth = []
    fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])
    assert any(h == "Bearer test-secret-value"
               for h in _KeyedLlama.seen_auth), _KeyedLlama.seen_auth


def test_no_key_configured_means_no_header_sent(keyed, monkeypatch):
    """Control: the tool must not invent credentials."""
    _clear_keys(monkeypatch)
    _KeyedLlama.seen_auth = []
    fleet_probe.probe_all(lanes=[("keyed", keyed, "llamacpp")])
    assert _KeyedLlama.seen_auth, "the keyed endpoints were never requested"
    assert all(h == "" for h in _KeyedLlama.seen_auth), (
        "an Authorization header was sent with no key configured: %r"
        % (_KeyedLlama.seen_auth,))


def test_a_rendered_keyed_lane_does_not_print_the_knob_hint(monkeypatch):
    """Control: once a key IS configured, naming the knob would be noise that
    implies the operator still has an action to take."""
    _clear_keys(monkeypatch)
    rows = [{
        "lane": "read", "url": "http://127.0.0.1:1", "reachable": True,
        "models": ["a"], "model_count": 1,
        "resident": ["a"], "resident_count": 1, "serving": True,
        "resident_error": None, "slots": (4, 0), "slots_error": None,
        "key_source": SHARED_KEY_VAR,
        "latency_ms": 1, "error": None,
    }]
    out = _render(monkeypatch, rows)
    assert "no API key configured" not in out, out
