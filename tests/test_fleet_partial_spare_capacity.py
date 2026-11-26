"""A server with SOME slots free is spare capacity, and the summary could not say so.

MEASURED 2026-11-29 against the live fleet, and reproduced below on a synthetic
server that answers the same three endpoints:

    3090-defiant  /health 200   /slots 200 -> 4 slots, 0 busy
                  fleet_probe prints   "1 loaded local server(s) with every slot
                                        free"                        <- seen

    same server,  /slots 200 -> 4 slots, 1 busy  (3 free)
                  fleet_probe prints   nothing at all              <- the defect

The spare predicate is ``r["slots"][0] > 0 and r["slots"][1] == 0``: it demands
that EVERY slot be free. So the day a local server takes its first concurrent
request, its remaining idle capacity stops being listed anywhere in the report,
and a router reading the summary sees a lane indistinguishable from a saturated
one. That is the exact fall-through to a billed lane this tool exists to
prevent, one slot width in.

WHY PARTIAL IS THE SAME CLAIM AS ALL-FREE, NOT A WEAKER ONE
------------------------------------------------------------
The all-free line is trustworthy because "the model is already resident, so it
cannot fail to load" -- the reason ``test_fleet_capacity_not_claimed.py`` refuses
to promote a *listing-only* probe to spare capacity. A server holding a model
with one slot busy satisfies that guarantee exactly as well as one with none:
the residency fact is unchanged, so free slots on it are real capacity. This file
does not touch the host-capacity line, which is about whether an IDLE daemon
could load anything -- a different question with a different answer.

WHAT IS NOT WEAKENED
--------------------
A 401 lane is still neither full nor partial: ``/slots`` was never read, so it
gets no free-slot claim of any kind (arm 6). A saturated lane still says nothing
about free slots (arm 5, a control). And a single-slot server that is free stays
on the all-free line rather than appearing twice (arm 7).
"""
import importlib.util
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMS = ROOT / "comms-lane"
sys.path.insert(0, str(COMMS))

import fleet_probe  # noqa: E402

PREFIX_FILE = "comms-lane/fleet_probe.py"

# The commit that still HAS the defect. Pinned by sha, never read as ``HEAD``:
# committing the fix moves HEAD onto the fixed tree, which leaves the control
# nothing to fail on and turns it red for the wrong reason. That happened here --
# suite red at e9fafb5 with ``the pre-fix tree already named partial capacity``,
# which was HEAD talking about the commit that fixed it.
PREFIX_SHA = "40c742438ac52d3dba17ea1d23e80b3cf7982337"


# --------------------------------------------------------------------------
# A llama.cpp-shaped server whose slot list the test sets directly. Class
# attributes, mutated between tests -- the constructor pattern every other
# fleet suite here uses, so http.server's own __init__ stays intact.
# --------------------------------------------------------------------------
class _LlamaCpp(BaseHTTPRequestHandler):
    slots = [{"is_processing": False}] * 4
    slots_status = 200
    health_status = 200
    models_status = 200
    seen = []

    def do_GET(self):  # noqa: N802 - http.server API
        cls = type(self)
        cls.seen.append(self.path)
        if self.path == "/health":
            code = cls.health_status
            body = {"status": "ok"} if code == 200 else {"error": "loading"}
        elif self.path == "/slots":
            code = cls.slots_status
            body = (cls.slots if code == 200
                    else {"error": {"message": "Invalid API Key", "code": 401}})
        elif self.path == "/v1/models":
            code = cls.models_status
            body = ({"object": "list", "data": [{"id": "m"}]}
                    if code == 200
                    else {"error": {"message": "Invalid API Key", "code": 401}})
        else:
            code, body = 404, {"error": "not found"}
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def llama():
    """A fresh llama.cpp server, defaults restored after every test."""
    _LlamaCpp.slots = [{"is_processing": False}] * 4
    _LlamaCpp.slots_status = 200
    _LlamaCpp.health_status = 200
    _LlamaCpp.models_status = 200
    _LlamaCpp.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _LlamaCpp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _render(monkeypatch, module, lanes):
    """Run ``module.main()`` over synthetic lanes and return what it printed.

    ``probe_all(lanes=LANES)`` binds its default at DEFINITION time, so patching
    the module attribute ``LANES`` leaves the real fleet in place. Patch the
    function -- the first draft of this helper in the sibling suite did the
    former and every arm silently measured the live Runner.
    """
    monkeypatch.setattr(module, "probe_all",
                        lambda *a, **k: [module.probe(label, url, kind)
                                         for label, url, kind in lanes])
    out = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: out.append(
                            " ".join(str(x) for x in a)))
    module.main([])
    return "\n".join(out)


# --------------------------------------------------------------------------
# THE GUARD: partial capacity must be named.
# --------------------------------------------------------------------------

def test_summary_names_partial_spare_capacity(monkeypatch, llama):
    """THE GUARD. Three of four slots free is capacity, and must be printed."""
    _LlamaCpp.slots = [{"is_processing": True}] + [{"is_processing": False}] * 3
    out = _render(monkeypatch, fleet_probe, [("part", llama, "llamacpp")])
    assert "partial capacity" in out, (
        "a loaded server with 3 of 4 slots free matched no line in the "
        "report, so a router reads it as saturated:\n" + out)
    assert "3/4 free" in out, (
        "the partial line must carry the number, not just the word:\n" + out)


def test_partial_lane_is_not_described_as_full(monkeypatch, llama):
    """The complement: partial capacity is not the all-free claim."""
    _LlamaCpp.slots = [{"is_processing": True}] + [{"is_processing": False}] * 3
    out = _render(monkeypatch, fleet_probe, [("part", llama, "llamacpp")])
    assert "every slot free" not in out, (
        "a server with one busy slot was announced as having every slot "
        "free:\n" + out)


def test_lane_line_already_carries_the_busy_count(monkeypatch, llama):
    """The per-lane line was already honest; this arm bounds the fix to the
    summary so a later refactor cannot move the number out of the lane row."""
    _LlamaCpp.slots = [{"is_processing": True}] + [{"is_processing": False}] * 3
    out = _render(monkeypatch, fleet_probe, [("part", llama, "llamacpp")])
    assert "1/4 slots busy" in out, (
        "the lane row stopped stating its busy count:\n" + out)


# --------------------------------------------------------------------------
# CONTROLS: must pass on BOTH trees, so the guards are not satisfied by an
# instrument that simply prints the new word for everything.
# --------------------------------------------------------------------------

def test_control_all_free_still_says_every_slot_free(monkeypatch, llama):
    out = _render(monkeypatch, fleet_probe, [("all", llama, "llamacpp")])
    assert "every slot free" in out, out
    assert "0/4" in out, out


def test_control_saturated_server_is_never_called_free(monkeypatch, llama):
    """Every slot busy: no free-slot claim of either kind."""
    _LlamaCpp.slots = [{"is_processing": True}] * 4
    out = _render(monkeypatch, fleet_probe, [("busy", llama, "llamacpp")])
    assert "every slot free" not in out, out
    assert "partial capacity" not in out, (
        "a saturated server was announced as partial capacity:\n" + out)


def test_control_unread_lane_is_not_partial_capacity(monkeypatch, llama):
    """A 401 is 'could not ask', never 'some slots free'."""
    _LlamaCpp.slots_status = 401
    out = _render(monkeypatch, fleet_probe, [("gated", llama, "llamacpp")])
    assert "partial capacity" not in out, (
        "a lane whose /slots was never read was announced as partial "
        "capacity:\n" + out)
    assert "UNKNOWN" in out, out


def test_control_single_slot_free_server_stays_on_the_all_free_line(
        monkeypatch, llama):
    """A 1-slot free server is all-free, not partial -- and must not be both."""
    _LlamaCpp.slots = [{"is_processing": False}]
    out = _render(monkeypatch, fleet_probe, [("one", llama, "llamacpp")])
    assert "every slot free" in out, out
    assert "partial capacity" not in out, out


def test_control_idle_lane_is_not_capacity(monkeypatch, llama):
    """A daemon with nothing loaded is still not spare capacity."""
    _LlamaCpp.health_status = 503
    out = _render(monkeypatch, fleet_probe, [("loading", llama, "llamacpp")])
    assert "partial capacity" not in out, out
    assert "every slot free" not in out, out


# --------------------------------------------------------------------------
# NEGATIVE CONTROL: the pre-fix module, from the tip, imported on its own.
# --------------------------------------------------------------------------
def _prefix_module():
    """Materialise ``fleet_probe`` from the PRE-FIX sha into its own directory.

    The copy lives OUTSIDE the repo so ``sys.path`` cannot fall back to the
    fixed module -- an editable install or a cwd insert would otherwise let the
    negative control measure the fixed tree twice. ``fleet_probe`` imports only
    the stdlib, so one file is a whole tree here.
    """
    import subprocess
    d = pathlib.Path(tempfile.mkdtemp(prefix="fleet-prefix-"))
    src = subprocess.run(["git", "-C", str(ROOT), "show", PREFIX_SHA + ":" + PREFIX_FILE],
                         capture_output=True, text=True, check=True).stdout
    (d / "fleet_probe.py").write_text(src, encoding="utf-8", newline="")
    spec = importlib.util.spec_from_file_location(
        "fleet_probe_prefix", d / "fleet_probe.py")
    # The pinned tree must actually BE the pre-fix tree. If a rewrite, a
    # rebase or a bad sha ever moved it onto the fixed side, every arm below
    # would go red claiming the defect is present -- the failure this whole
    # pin exists to stop being silent. Fail here instead, naming the cause.
    assert "partial capacity" not in src, (
        "PREFIX_SHA %s already contains the fix, so it is not the pre-fix tree "
        "and every arm below would fail for the wrong reason" % PREFIX_SHA[:12])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(module, slots):
    return {
        "lane": "part", "url": "http://example.invalid", "reachable": True,
        "models": ["m"], "model_count": 1, "resident": ["m"],
        "resident_count": 1, "serving": True,
        "resident_error": None, "slots": slots, "slots_error": None,
        "key_source": None, "latency_ms": 1, "error": None,
    }


def test_negative_control_prefix_tree_drops_partial_capacity(monkeypatch):
    """Red on the pre-fix tree, green here -- the guard is load-bearing.

    Measured by VALUE, not by the absence of a new token: the pre-fix tree is
    shown to print NOTHING for a 3-of-4-free server while the lane row beside it
    states the busy count, so the defect is droppage and not a wording choice.
    """
    pre = _prefix_module()
    row = _row(pre, (4, 1))

    out = []
    monkeypatch.setattr(pre, "probe_all", lambda *a, **k: [row])
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: out.append(" ".join(str(x) for x in a)))
    pre.main([])
    old = "\n".join(out)

    # The defect, restated on the tree that has it.
    assert "partial capacity" not in old, (
        "the pre-fix tree already named partial capacity, so this control "
        "bounds nothing:\n" + old)
    assert "free slots" not in old, (
        "the pre-fix tree already named free slots:\n" + old)
    # ...and it is not silent for lack of information: the lane row carries
    # the numbers, so the summary is dropping a fact it was handed.
    assert "1/4 slots busy" in old, (
        "the pre-fix tree lacked the busy count too, so this is a renderer "
        "gap and not the summary predicate:\n" + old)

    # Same row, fixed tree: named, with the number.
    monkeypatch.undo()
    out2 = []
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: [row])
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: out2.append(" ".join(str(x) for x in a)))
    fleet_probe.main([])
    new = "\n".join(out2)
    assert "partial capacity" in new, (
        "the fixed tree did not name partial capacity for the same row:\n" + new)
    assert "3/4 free" in new, new


def test_negative_control_prefix_tree_still_counts_all_free(monkeypatch):
    """The all-free line is unchanged on both trees -- the control's control."""
    pre = _prefix_module()
    row = _row(pre, (4, 0))
    out = []
    monkeypatch.setattr(pre, "probe_all", lambda *a, **k: [row])
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: out.append(" ".join(str(x) for x in a)))
    pre.main([])
    old = "\n".join(out)
    assert "every slot free" in old, old
    assert "partial capacity" not in old, old
