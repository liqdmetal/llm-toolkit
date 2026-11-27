"""A slot holding a task is not evidence of freedom, and the spare-capacity line
certified it as free anyway.

`read_slots` counts a slot as busy only when `is_processing` or
`is_processing_prompt` is set. The summary then prints, for a lane with zero
busy:

    [1] loaded local server(s) with every slot free (MEASURED spare capacity:
        3090-defiant 0/4, n_ctx 4096)

MEASURED 2026-12-04 against the live fleet on the one lane that line certified.
Nine reads of `3090-defiant` (127.0.0.1:8090) across ~40 seconds, every read
identical:

    slot0-2  is_processing false, id_task None
    slot3    is_processing false, id_task 158, n_prompt_tokens 19,
             n_prompt_tokens_processed 0

So the report's single certified-free lane was holding an assigned task with
prompt tokens ingested and none advanced, while the summary said EVERY slot was
free. That is this repo's standing class in its sharpest form: not a zero
standing in for an absence, but a MAXIMUM standing in for an unmeasured slot.
The busy flags state what a slot is doing; they do not state whether it is free.

WHAT THE FIX DOES, AND WHAT IT REFUSES TO DO
-------------------------------------------
`read_slot_detail` carries a fourth fact out of the SAME single `/slots` read --
the module forbids a second read ("two reads of a live server can disagree with
each other") -- and the summary states it beside the figure it qualifies:

    3090-defiant 0/4, n_ctx 4096 [slot 3 holds task 158 -- not counted busy,
    not counted free]

It is deliberately NOT counted as busy. Whether that task is reserved work about
to prefill or state left behind by a finished one cannot be decided from the
/slots document, and choosing either answer would be the same defect facing the
other way. So the number and its qualifier travel together, which is the same
repair `_ctx_suffix` made for n_ctx scope and `host_capacity` made for daemons
vs machines.

A lane with no parked slot renders BYTE-IDENTICALLY to before. That is the
counter-arm, not a nicety: a qualifier that appears everywhere says nothing.

HOW THE ARMS ARE SPLIT
----------------------
  TestReadControl    read_slot_detail's parked detection. Fails pre-fix BY
                     VALUE through a tolerant accessor (None, not AttributeError),
                     so the arm grades behaviour rather than a symbol's absence.
  TestRenderControl  the summary must state the qualification. Driven through
                     main() with a hand-built row, so BOTH trees receive the
                     same input and the difference is the rendered sentence.
  TestCounterArms    must stay GREEN on BOTH trees: no parked slot means no
                     qualifier, a busy slot is busy and not parked, and the
                     lane line's busy count does not move.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comms-lane"))

import fleet_probe  # noqa: E402

PREFIX_FILE = "comms-lane/fleet_probe.py"

#: The last commit BEFORE this fix. Pinned by sha, never read as HEAD: committing
#: the fix moves HEAD onto the fixed tree, which would leave the control nothing
#: to fail on and turn it red for the wrong reason.
PREFIX_SHA = "59267b4b6396045899ccb7aaa6361fecb3771fcb"


# --------------------------------------------------------------------------
# A llama.cpp-shaped server whose slot list the test sets directly.
# --------------------------------------------------------------------------
class _LlamaCpp(BaseHTTPRequestHandler):
    slots = [{"is_processing": False, "n_ctx": 4096}] * 4
    slots_status = 200

    def do_GET(self):  # noqa: N802 - http.server API
        cls = type(self)
        if self.path == "/health":
            code, body = 200, {"status": "ok"}
        elif self.path == "/slots":
            code = cls.slots_status
            body = (cls.slots if code == 200
                    else {"error": {"message": "Invalid API Key", "code": 401}})
        elif self.path == "/v1/models":
            code, body = 200, {"object": "list", "data": [{"id": "m"}]}
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
    _LlamaCpp.slots = [{"is_processing": False, "n_ctx": 4096}] * 4
    _LlamaCpp.slots_status = 200
    srv = HTTPServer(("127.0.0.1", 0), _LlamaCpp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


PARKED_SLOT = {"id": 3, "is_processing": False, "is_processing_prompt": False,
               "n_ctx": 4096, "id_task": 158, "n_prompt_tokens": 19,
               "n_prompt_tokens_processed": 0}
FREE_SLOT = {"id": 0, "is_processing": False, "n_ctx": 4096}


def _parked_of(module, doc):
    """Parked slots from one /slots document, tolerant across trees.

    Returns None where `read_slot_detail` does not exist, so every arm below
    fails BY VALUE on the pre-fix tree instead of dying on a missing symbol. A
    control that cannot run proves nothing about the behaviour.
    """
    reader = getattr(module, "read_slot_detail", None)
    if reader is None:
        return None
    real = module._get
    module._get = lambda url=None, timeout=None, key=None, _d=doc: _d
    try:
        return reader("http://127.0.0.1:1", "llamacpp")["parked"]
    finally:
        module._get = real


# --------------------------------------------------------------------------
# THE READ GUARD.
# --------------------------------------------------------------------------
class TestReadControl:
    def test_a_clear_busy_flag_with_a_held_task_is_parked(self):
        doc = [FREE_SLOT, FREE_SLOT, FREE_SLOT, PARKED_SLOT]
        parked = _parked_of(fleet_probe, doc)
        assert parked is not None, "read_slot_detail is absent on this tree"
        assert len(parked) == 1, (
            "one slot held an assigned task while reporting not-busy; the "
            "reader reported %r" % (parked,))
        assert parked[0]["slot"] == 3
        assert parked[0]["id_task"] == 158

    def test_a_processing_slot_is_busy_and_not_parked(self):
        busy = dict(PARKED_SLOT, is_processing=True)
        parked = _parked_of(fleet_probe, [busy, FREE_SLOT])
        assert parked == [], (
            "a slot that is processing is already counted busy; parking it too "
            "double-counts the same slot: %r" % (parked,))

    def test_a_negative_task_id_is_not_a_held_task(self):
        """A build reporting -1 for an idle slot must not read as task 0."""
        idle = dict(FREE_SLOT, id_task=-1)
        parked = _parked_of(fleet_probe, [idle])
        assert parked == [], "id_task -1 read as a held task: %r" % (parked,)

    def test_the_parked_fact_comes_from_the_same_single_read(self):
        """One GET, not two: two reads of a live server can disagree."""
        calls = []
        real = fleet_probe._get
        fleet_probe._get = lambda url=None, timeout=None, key=None: (
            calls.append(url), [PARKED_SLOT])[1]
        try:
            fleet_probe.read_slot_detail("http://127.0.0.1:1", "llamacpp")
        finally:
            fleet_probe._get = real
        assert len(calls) == 1, (
            "the figure and its qualifier must come from ONE /slots read; got "
            "%d" % len(calls))


# --------------------------------------------------------------------------
# THE RENDER GUARD.
# --------------------------------------------------------------------------
def _row(slots=(4, 0), parked=None, lane="3090-defiant"):
    return {
        "lane": lane, "url": "http://example.invalid", "kind": "llamacpp",
        "reachable": True, "models": ["m"], "model_count": 1,
        "resident": ["m"], "resident_count": 1, "serving": True,
        "resident_error": None, "slots": slots, "slots_error": None,
        "slots_cause": None, "slot_ctx": 4096, "slot_ctx_slots": 4,
        "parked": parked, "key_source": None, "latency_ms": 1, "error": None,
    }


def _render(monkeypatch, module, row):
    monkeypatch.setattr(module, "probe_all", lambda *a, **k: [row])
    out = []
    monkeypatch.setattr("builtins.print",
                        lambda *a, **k: out.append(" ".join(str(x) for x in a)))
    module.main([])
    return "\n".join(out)


class TestRenderControl:
    def test_the_certified_lane_states_the_held_task(self, monkeypatch):
        parked = [{"slot": 3, "id_task": 158, "prompt_tokens": 19,
                   "processed": 0}]
        out = _render(monkeypatch, fleet_probe, _row(parked=parked))
        assert "spare capacity" in out, (
            "this arm is about qualifying the spare line; it was not printed:\n"
            + out)
        assert "holds task 158" in out, (
            "the spare-capacity line certified every slot free while slot 3 "
            "held task 158; the qualification is missing:\n" + out)
        assert "not counted busy, not counted free" in out, (
            "the note must refuse to call the slot either busy or free, "
            "because /slots cannot decide it:\n" + out)


class TestCounterArms:
    """Must stay GREEN on BOTH trees."""

    def test_a_lane_with_no_held_task_gets_no_qualifier(self, monkeypatch):
        out = _render(monkeypatch, fleet_probe, _row(parked=[]))
        assert "spare capacity" in out
        assert "holds task" not in out, (
            "a qualifier that prints everywhere says nothing:\n" + out)
        assert "not counted" not in out

    def test_an_unread_parked_state_gets_no_qualifier(self, monkeypatch):
        """None means the read never happened, which is not a held task."""
        out = _render(monkeypatch, fleet_probe, _row(parked=None))
        assert "spare capacity" in out
        assert "holds task" not in out

    def test_the_lane_line_still_carries_the_busy_count(self, monkeypatch):
        out = _render(monkeypatch, fleet_probe, _row(parked=[]))
        assert "0/4 slots busy" in out, (
            "the busy count is a different fact and must not move:\n" + out)

    def test_a_busy_slot_is_still_never_called_free(self, monkeypatch):
        out = _render(monkeypatch, fleet_probe,
                      _row(slots=(4, 1), parked=[]))
        assert "spare capacity" not in out, (
            "a saturated server rendered as spare capacity:\n" + out)


# --------------------------------------------------------------------------
# NEGATIVE CONTROL: the pinned pre-fix tree.
# --------------------------------------------------------------------------
def _prefix_module():
    """Materialise `fleet_probe` from the PRE-FIX sha into its own directory.

    Outside the repo so sys.path cannot fall back to the fixed module, which
    would make the control measure the fixed tree twice.
    """
    d = pathlib.Path(tempfile.mkdtemp(prefix="fleet-parked-prefix-"))
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", PREFIX_SHA + ":" + PREFIX_FILE],
        capture_output=True, text=True, check=True).stdout
    (d / "fleet_probe.py").write_text(src, encoding="utf-8", newline="")
    spec = importlib.util.spec_from_file_location(
        "fleet_probe_parked_prefix", d / "fleet_probe.py")
    assert "parked" not in src, (
        "PREFIX_SHA %s already carries the fix, so it is not the pre-fix tree "
        "and this control bounds nothing" % PREFIX_SHA[:12])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_control_prefix_tree_certifies_a_held_task_as_free(monkeypatch):
    """Red on the pre-fix tree, green here -- by VALUE, not by absence.

    The same hand-built row goes to both trees, so the only difference is the
    sentence each renders. The pre-fix tree is shown to print the spare line
    WITHOUT the qualification, which is the defect itself and not a renderer
    gap: it had the numbers and did not state the fact that decides their use.
    """
    pre = _prefix_module()
    parked = [{"slot": 3, "id_task": 158, "prompt_tokens": 19, "processed": 0}]
    row = _row(parked=parked)

    old = _render(monkeypatch, pre, row)
    assert "spare capacity" in old, (
        "the pre-fix tree did not print the spare line at all, so this control "
        "measures a renderer gap and not the missing qualification:\n" + old)
    assert "holds task" not in old, (
        "the pre-fix tree already stated the held task:\n" + old)
    assert "0/4" in old, (
        "the pre-fix tree lacked the slot figures too:\n" + old)

    monkeypatch.undo()
    new = _render(monkeypatch, fleet_probe, row)
    assert "holds task 158" in new, (
        "the fixed tree did not qualify the same row:\n" + new)
