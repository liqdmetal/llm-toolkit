"""A loaded llama.cpp lane on the 3090 must not be invisible to the roster.

THE FINDING, measured 2026-11-28 before this file existed
--------------------------------------------------------
``fleet_probe.py`` printed::

    3090  UP    7 installed, IDLE 0/7 loaded, 26 ms

for a box that was holding a model. A second llama.cpp server answers there,
keyless, through an ssh tunnel this roster did not know about::

    $ netstat -ano | grep :8090
      TCP  127.0.0.1:8090  LISTENING  7824
    $ (Get-CimInstance Win32_Process -Filter 'ProcessId=7824').CommandLine
      ssh -fN -i ...\\a3090_ed25519 -L 8090:127.0.0.1:8090 a3090@192.168.1.124

    /health     200 {"status":"ok"}
    /v1/models  200 -> /home/a3090/mm/defiant-vision.gguf
    /slots      4 slots, 0 busy, n_ctx 4096 each

``LANES`` listed that host's ollama daemon and its 18082 Inkling tunnel, and
nothing else. So the probe asked two daemons and printed the answer as the
box's state -- for the THIRD time on a third daemon:

  2026-11-26  5090: ollama idle beside llama.cpp serving      (test_fleet_llamacpp_lane)
  2026-11-28  3090: ollama idle beside Inkling on 18082       (test_fleet_3090_tunnel_lane)
  2026-11-28  3090: ollama idle beside defiant-vision on 8090 (this file)

WHY IT IS THE EXPENSIVE DIRECTION
---------------------------------
This tool's own header: "a reachable-but-idle lane is the expensive default ...
because the router falls through to a billed one." An unrostered SERVING lane
does not merely under-report capacity -- it reports a busy box as free, so the
router sends work to a billed lane while a local card holds a loaded model.

Two facts must hold, and they are separate:
  1. ROSTER -- the lane is in ``LANES`` at all. A row nobody asks about cannot
     be rendered, whatever the renderer does.
  2. ATTRIBUTION -- the tunnel is mapped to its FAR END. Without it the 3090 is
     counted THREE times: idle on its ollama daemon, serving Inkling on 18082,
     serving defiant-vision on 8090. "One host = one GPU" stops being true in
     the direction that overstates spare capacity.

GUARDS FAIL ON THE PRE-FIX TREE BY VALUE, NOT BY A MISSING SYMBOL
-----------------------------------------------------------------
``LANES``, ``TUNNEL_HOSTS``, ``_host_of`` and ``host_capacity`` all exist on the
pre-fix tree, so these arms execute unchanged there and fail on their
assertions -- a lane list missing an entry, and a host count of 2 where 1 is
true. No arm names anything the fix introduces.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import fleet_probe  # noqa: E402

H5090 = "http://192.168.1.232:11434"    # ollama on the 5090 host
HQ38 = "http://192.168.1.232:8081"      # llama.cpp on the SAME host
H3090 = "http://192.168.1.124:11434"    # ollama on the 3090 host
INKLING = "http://127.0.0.1:18082"      # tunnel -> 3090, llama.cpp
DEFIANT = "http://127.0.0.1:8090"       # tunnel -> 3090, llama.cpp


def _row(lane, url, *, hosted=False, resident=(), resident_error=None,
         model_count=3):
    """One probe row in exactly the shape ``probe()`` returns."""
    if hosted or resident_error is not None:
        resident_list, resident_count, serving = None, None, None
    else:
        resident_list = list(resident)
        resident_count = len(resident_list)
        serving = bool(resident_list)
    return {
        "lane": lane, "url": url, "reachable": True,
        "models": None if model_count is None else ["m"] * model_count,
        "model_count": model_count,
        "resident": resident_list, "resident_count": resident_count,
        "serving": serving, "resident_error": resident_error,
        "latency_ms": 1, "error": None,
    }


def _run(rows, monkeypatch):
    monkeypatch.setattr(fleet_probe, "probe_all", lambda *a, **k: rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fleet_probe.main([])
    assert rc == 0, "a down lane is data; main() must still exit 0"
    return buf.getvalue()


def _host_line(out):
    for line in out.splitlines():
        if "local host" in line:
            return line
    return None


# ------------------------------------------------------------------ guards ---

def test_the_defiant_lane_is_actually_in_the_roster():
    """GUARD 1, ROSTER. A lane nobody asks about cannot be rendered.

    Pre-fix, ``LANES`` had five entries and none of them was this server, so the
    strongest possible renderer change would still have reported nothing about
    it. The URL is asserted, not the label: a lane can be renamed, but the
    endpoint is what makes it the same server.
    """
    urls = [url.rstrip("/") for _label, url, _kind in fleet_probe.LANES]
    assert DEFIANT in urls, (
        "the 3090's defiant-vision server answers on %s with a model loaded and "
        "appears in NO lane; roster is %r" % (DEFIANT, urls))


def test_the_defiant_lane_is_asked_the_llamacpp_question():
    """GUARD 1b. Rostering it with the wrong kind reports a false absence.

    ``ollama`` kind would ask /api/tags -- a different daemon on a different
    port would answer, and the answer would describe a box the lane is not on.
    """
    kinds = {url.rstrip("/"): kind for _l, url, kind in fleet_probe.LANES}
    assert kinds.get(DEFIANT) == "llamacpp", (
        "expected llamacpp for %s, roster says %r" % (DEFIANT, kinds.get(DEFIANT)))


def test_both_3090_tunnels_are_attributed_to_the_far_end():
    """GUARD 2, ATTRIBUTION. A tunnel answers on loopback; it MEASURES a remote.

    Pre-fix, ``_host_of`` returned "127.0.0.1" for 8090, so the 3090 became two
    hosts -- one of them a loopback that looks like spare local capacity.
    """
    assert fleet_probe._host_of(INKLING) == "192.168.1.124", (
        "the Inkling tunnel must be attributed to the 3090, got %r"
        % (fleet_probe._host_of(INKLING),))
    assert fleet_probe._host_of(DEFIANT) == "192.168.1.124", (
        "the defiant-vision tunnel must be attributed to the 3090, got %r -- a "
        "loopback host here counts one GPU as two machines"
        % (fleet_probe._host_of(DEFIANT),))


def test_three_daemons_on_the_3090_are_one_machine(monkeypatch):
    """GUARD 3, the consequence: the host count, through the rendered line.

    The real fleet shape. Pre-fix the line reads ``[0/3] local hosts`` because
    loopback 8090 is its own "host" -- and a count that runs HIGH is the
    expensive direction: more advertised room than exists.
    """
    rows = [
        _row("nous", "http://127.0.0.1:8099", hosted=True),
        _row("5090", H5090, resident=[], model_count=3),
        _row("q38", HQ38, resident=["<loaded; name key-gated>"], model_count=None),
        _row("3090", H3090, resident=[], model_count=7),
        _row("3090-inkling", INKLING, resident=["<loaded; name key-gated>"],
             model_count=None),
        _row("3090-defiant", DEFIANT, resident=["<loaded; name key-gated>"],
             model_count=None),
    ]
    line = _host_line(_run(rows, monkeypatch))
    assert line is not None, "no local-host capacity line was printed"
    assert "[0/2]" in line, (
        "two machines exist (5090 host and 3090 host) and neither is free; "
        "line says: %r" % (line,))


# -------------------------------------------------------------- the controls ---

def test_a_genuinely_separate_host_is_still_separate(monkeypatch):
    """CONTROL. Without this the fix passes by mapping everything to one host."""
    rows = [
        _row("5090", H5090, resident=[], model_count=3),
        _row("3090", H3090, resident=[], model_count=7),
    ]
    line = _host_line(_run(rows, monkeypatch))
    assert "[2/2]" in line, (
        "two real machines with nothing serving is two units of capacity; "
        "line says: %r" % (line,))


def test_an_unmapped_loopback_lane_is_not_silently_merged():
    """CONTROL, the other direction: attribution is a MAP, not a wildcard.

    A loopback URL nobody mapped must be its own host. Otherwise "attribute the
    far end" could be implemented as "call every 127.0.0.1 the 3090", which
    would merge unrelated local services into one GPU.
    """
    assert fleet_probe._host_of("http://127.0.0.1:9999") == "127.0.0.1", (
        "an unmapped loopback lane must keep its own identity")


def test_every_tunnelled_lane_names_a_far_end_that_is_a_real_lane():
    """CONTROL. A TUNNEL_HOSTS entry pointing at a host no lane lives on would
    silently create a phantom machine in the denominator."""
    lane_hosts = {fleet_probe._host_of(url) for _l, url, _k in fleet_probe.LANES}
    for tunnel_url, far_end in fleet_probe.TUNNEL_HOSTS.items():
        assert far_end in lane_hosts, (
            "%s is attributed to %r, which no lane reports -- that is a phantom "
            "host in denominator" % (tunnel_url, far_end))
