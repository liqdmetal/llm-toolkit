"""A lane that never answered must not report the same figure as a lane with nothing installed.

THE DEFECT, MEASURED 2026-12-03 at 297081a
------------------------------------------
``probe()``'s failure path built a row that claimed the lane had ANSWERED::

    "reachable": False,
    "models": [],
    "model_count": 0,

Meantime a live lane answering ``{"models": []}`` produced those same two values.
Measured by running the real ``probe()`` against (a) a closed port and (b) a
server answering an empty model list::

    DOWN       reachable=False  models=[]  model_count=0  error='URLError: ...'
    UP_EMPTY   reachable=True   models=[]  model_count=0  error=None

``models`` and ``model_count`` were byte-identical, so ``--json`` -- the
machine-readable surface -- could not tell a dead lane from an empty live one.
The human report escaped it only because it prints ``r["error"]`` and
``continue``s before reading either key. That is ABSENT-vs-ZERO in the one place
a consumer has no guard of its own.

It also broke the module's OWN rule, three functions above, in ``list_models``:
"Returns None where the lane CANNOT answer the question, which is different from
[] (it answered, nothing installed)." The failure path was the single branch
that disobeyed it.

SECOND HALF -- the lane vanished from capacity accounting
--------------------------------------------------------
``host_capacity()`` skipped rows on::

    if r["resident"] is None and r["resident_error"] is None:
        continue  # hosted: no residency notion, not part of local capacity

Both Nones are TRUE for a HOSTED lane (the question does not apply) and TRUE for
a DOWN lane (nothing was read because nothing answered). One predicate, two
opposite facts: a dead local lane was skipped as if it were hosted, and its HOST
then appeared in NO line of the report -- not free, not shared, not unread.
Measured 2026-12-03: a fleet of one live hosted lane plus two dead local lanes
rendered with no local-capacity line at all, i.e. smaller and healthier than the
truth.

A THIRD STATE, pinned so the fix cannot become "always report None"
------------------------------------------------------------------
A lane that ANSWERED with an empty listing must still report a measured 0 and an
empty list. If that arm went green by turning everything into None, the report
would have traded a fabricated absence for a fabricated unknown, and the sqlite
answer for "is anything installed on this box" would be lost.

WHAT IS ASSERTED
----------------
Behaviour of the REAL ``probe()`` and the REAL ``host_capacity()``. Not a key
the fix invents: a control that asserts on a newly added dict key fails on the
pre-fix tree by ``KeyError`` and bounds nothing. Every arm here reads values
that both trees produce.
"""
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMS = ROOT / "comms-lane"
sys.path.insert(0, str(COMMS))

import fleet_probe  # noqa: E402

PREFIX_FILE = "comms-lane/fleet_probe.py"

#: The commit that still HAS the defect. An EXPLICIT sha, never ``HEAD``:
#: committing this fix moves HEAD onto the fixed tree and leaves every control
#: below nothing to fail on, which turns them red for the wrong reason. That
#: exact mistake was mine in roster ``237e5d2`` and it is why this is pinned.
PREFIX_SHA = "297081ae3fa4d6bb9aec545336bf473307278453"


# ---------------------------------------------------------------------------
# A server whose model list the test sets, so "answered with nothing" is a real
# HTTP answer and not a hand-built dict. http.server's own __init__ is left
# alone; the body is a class attribute mutated between tests, the pattern every
# fleet suite in this repo uses.
# ---------------------------------------------------------------------------
class _Listing(BaseHTTPRequestHandler):
    body = b'{"models": []}'

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


@pytest.fixture
def empty_lane():
    """A live ollama-shaped server that answers with zero models installed."""
    _Listing.body = b'{"models": []}'
    srv = HTTPServer(("127.0.0.1", 0), _Listing)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _dead_port():
    """A port nothing is listening on -- bound, read, then released."""
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _down_row(module=fleet_probe):
    return module.probe("dead", "http://127.0.0.1:%d" % _dead_port(),
                        "ollama", timeout=1.0)


def _render(module, rows):
    """The human report for ``rows``, with the probe stubbed out."""
    saved, out = module.probe_all, io.StringIO()
    module.probe_all = lambda lanes=None, timeout=None: rows
    try:
        with redirect_stdout(out):
            module.main([])
    finally:
        module.probe_all = saved
    return out.getvalue()


def _row(lane, url, reachable=True, resident=None, resident_error=None,
         models=None, model_count=None, serving=None, error=None):
    """A row shaped exactly like probe()'s, for the capacity arms."""
    return {
        "lane": lane, "url": url, "reachable": reachable,
        "models": models, "model_count": model_count,
        "resident": resident,
        "resident_count": None if resident is None else len(resident),
        "serving": (serving if serving is not None
                    else (None if resident is None else bool(resident))),
        "resident_error": resident_error,
        "slots": None, "slots_error": None, "slots_cause": None,
        "slot_ctx": None, "key_source": None, "latency_ms": 1, "error": error,
    }


# ---------------------------------------------------------------------------
# GUARDS: green on the fixed tree, RED on PREFIX_SHA.
# ---------------------------------------------------------------------------
def test_a_down_lane_reports_unknown_not_a_measured_zero():
    """The load-bearing arm: a closed port must not answer "0 models"."""
    row = _down_row()
    assert row["reachable"] is False
    assert row["model_count"] is None, (
        "a lane that never answered reported model_count=%r, which is the same "
        "figure a live lane with nothing installed reports"
        % (row["model_count"],))
    assert row["models"] is None, row["models"]
    assert row["error"], "a down lane must still say WHY it could not be asked"


def test_a_down_row_and_an_up_row_carry_the_same_keys(empty_lane):
    """Schema parity, asserted by VALUE so it fails on the pre-fix tree.

    A consumer reading ``row["slots_cause"]`` is reading a key that was present
    on every answered row and ABSENT on every down row -- so it raised on
    exactly the lanes it most needed to explain. Compared as sets rather than
    by subscripting, because a KeyError would prove only that a name was added.
    """
    down = _down_row()
    up = fleet_probe.probe("up", empty_lane, "ollama", timeout=1.0)
    assert up["reachable"] is True, up["error"]
    assert set(down) == set(up), (
        "down-only keys %r / up-only keys %r -- a row's shape must not depend "
        "on whether the lane happened to answer"
        % (sorted(set(down) - set(up)), sorted(set(up) - set(down))))


def test_an_unreachable_lane_makes_its_host_unread_not_absent():
    """A dead local lane must reach unread_hosts, not be skipped as hosted."""
    rows = [_row("5090", "http://192.168.1.232:11434", reachable=False,
                 error="URLError: timed out")]
    by_host, free, shared, unread = fleet_probe.host_capacity(rows)
    assert "192.168.1.232" in by_host, (
        "an unreachable local lane was dropped from capacity accounting "
        "entirely, so its host appeared in no line of the report")
    assert "192.168.1.232" in unread, (
        "a host whose daemon never answered is UNKNOWN, not free: %r" % (unread,))
    assert "192.168.1.232" not in free, (
        "an unreachable host must never be certified free")


def test_the_report_names_an_unread_host_rather_than_going_quiet():
    """The rendered sentence -- the contract a router actually reads."""
    rows = [_row("nous", "http://127.0.0.1:8099", reachable=True, models=None,
                 model_count=None),
            _row("5090", "http://192.168.1.232:11434", reachable=False,
                 error="URLError: timed out")]
    out = _render(fleet_probe, rows)
    assert "residency unread on" in out or "unread" in out, (
        "a fleet holding a dead local lane rendered with no line naming it:\n"
        + out)


# ---------------------------------------------------------------------------
# CONTROLS: green on BOTH trees. These stop the fix from over-reaching.
# ---------------------------------------------------------------------------
def test_a_lane_that_answered_with_nothing_still_reports_a_measured_zero(
        empty_lane):
    """The third state. 0 means "asked, nothing installed" and must survive.

    If this went red, the fix would have replaced a fabricated absence with a
    fabricated unknown -- and "is anything installed on that box" would have no
    answer anywhere in the report.
    """
    row = fleet_probe.probe("up", empty_lane, "ollama", timeout=1.0)
    assert row["reachable"] is True, row["error"]
    assert row["model_count"] == 0, (
        "a lane that answered an empty listing reported %r, but it DID answer"
        % (row["model_count"],))
    assert row["models"] == [], row["models"]
    assert row["error"] is None


def test_a_hosted_lane_is_still_not_local_capacity():
    """The other half of the collapsed predicate: nous has no residency notion."""
    rows = [_row("nous", "http://127.0.0.1:8099", reachable=True, models=None,
                 model_count=None)]
    by_host, free, shared, unread = fleet_probe.host_capacity(rows)
    assert by_host == {}, (
        "a hosted lane is not a local host and must not enter capacity "
        "accounting: %r" % (by_host,))
    assert unread == []


def test_a_healthy_down_lane_still_renders_as_down_with_its_reason():
    """The human report must keep saying DOWN and naming the cause."""
    out = _render(fleet_probe, [_down_row()])
    assert "DOWN" in out, out
    assert "URLError" in out or "timed out" in out, out


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: the pre-fix module, materialised from the pinned sha into
# its own directory.
# ---------------------------------------------------------------------------
def _prefix_module():
    """Import ``fleet_probe`` from PREFIX_SHA, outside the repo.

    Outside, deliberately: an editable install or a cwd ``sys.path`` insert
    would let the control fall back to the FIXED module and measure it twice.
    ``fleet_probe`` is stdlib-only, so one file is a whole tree here.
    """
    d = pathlib.Path(tempfile.mkdtemp(prefix="fleet-down-prefix-"))
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", PREFIX_SHA + ":" + PREFIX_FILE],
        capture_output=True, text=True, check=True).stdout
    (d / "fleet_probe.py").write_text(src, encoding="utf-8", newline="")
    # The pinned tree must actually BE the pre-fix tree. If a rewrite, a rebase
    # or a bad sha ever moved it onto the fixed side, every control below would
    # go red claiming the defect is present -- the silent failure this pin
    # exists to stop. Fail here instead, naming the cause.
    assert '"models": [],' in src and '"models": None,' not in src, (
        "PREFIX_SHA %s is not the pre-fix tree (its down row already reports "
        "models=None), so the controls below would fail for the wrong reason"
        % PREFIX_SHA[:12])
    spec = importlib.util.spec_from_file_location(
        "fleet_probe_down_prefix", d / "fleet_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefix_tree_fabricates_a_zero_for_a_lane_that_never_answered():
    """RED on the pre-fix tree, GREEN here: the guard is load-bearing."""
    pre = _prefix_module()
    row = _down_row(pre)
    assert row["reachable"] is False
    assert row["model_count"] == 0, (
        "the pre-fix tree no longer fabricates a zero, so this control has "
        "nothing to measure: %r" % (row["model_count"],))
    assert row["models"] == [], row["models"]


def test_prefix_tree_loses_the_unreachable_host_from_capacity():
    """RED on the pre-fix tree: the host is skipped, not reported unread."""
    pre = _prefix_module()
    rows = [_row("5090", "http://192.168.1.232:11434", reachable=False,
                 error="URLError: timed out")]
    by_host, free, shared, unread = pre.host_capacity(rows)
    assert by_host == {}, (
        "the pre-fix tree now keeps an unreachable host in capacity "
        "accounting, so this control has nothing to measure: %r" % (by_host,))
    assert unread == []


def test_prefix_tree_renders_a_dead_fleet_with_no_local_line():
    """RED on the pre-fix tree: the local-capacity line is simply absent."""
    pre = _prefix_module()
    rows = [pre.probe("nous", "http://127.0.0.1:8099", "openai", timeout=1.0)]
    # Skip if nous is not actually answering on this run: a control that
    # depends on a network lane is a control that fails for the wrong reason.
    if not rows[0]["reachable"]:
        pytest.skip("nous is not answering this run; the arm needs one live "
                    "hosted lane and would otherwise measure an outage")
    rows = rows + [
        pre.probe("5090", "http://127.0.0.1:%d" % _dead_port(), "ollama",
                  timeout=1.0),
        pre.probe("3090", "http://127.0.0.1:%d" % _dead_port(), "ollama",
                  timeout=1.0),
    ]
    out = _render(pre, rows)
    assert "local" not in out, (
        "the pre-fix tree now emits a local-capacity line for a fleet holding "
        "two dead local lanes, so this control has nothing to measure:\n" + out)
