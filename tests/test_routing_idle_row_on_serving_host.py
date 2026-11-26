"""A policy row may not call a daemon "idle" while its own HOST is serving.

THE DEFECT THIS PINS
--------------------
`fleet_probe.py` already states the distinction the policy table then threw away.
Its capacity summary prints, in its own words::

    [0/2] local hosts with no daemon serving (capacity NOT established: ...)
          (idle daemons on a busy host: 5090, 3090)

...and the same run's table rows read::

    | 5090 ollama  ... | UP, idle 0/3 loaded | ... |
    | 3090 ollama  ... | UP, idle 0/7 loaded | ... |

Both are true statements about a DAEMON and a misleading statement about a
MACHINE. The 5090's ollama holds no model while q38's llama.cpp on the SAME card
serves one; the 3090's ollama holds no model while Inkling and defiant-vision on
the SAME card both serve. A reader who routes on "idle" dispatches to a busy GPU
— which is the expensive direction for the goal this stack exists to serve, and
the same defect `test_fleet_llamacpp_lane.py` fixed in the probe and
`test_fleet_capacity_not_claimed.py` fixed in the summary. The table was the last
surface still carrying it.

WHY THE CHECK IS STRUCTURAL, NOT A LIVE PROBE
---------------------------------------------
A guard that re-probed the fleet would go red whenever a lane went down, and a
doc guard must fail on the DOC's defect, not on the fleet's state. So this test
reads only the document: if two rows describe the SAME HOST and one of them
states SERVING, then the other may not claim idle capacity without saying that
the host is serving. No network, no GPU, no daemon.
"""

import pathlib
import re
import subprocess
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROUTING = ROOT / "comms-lane" / "ROUTING.md"

sys.path.insert(0, str(ROOT / "comms-lane"))
import fleet_probe  # noqa: E402

# The document BEFORE the fix. Pinned by SHA rather than by `!= HEAD`: a
# comparison against HEAD stops being a negative control the moment the fix is
# committed, because HEAD then moves onto the fixed tree and the arm can only
# fail. Measured by CONTENT at this commit, it reproduces the defect.
#
# MEASURED 2026-12-01, and this is my own second correction on this file: I first
# pinned 237e5d2 (the commit the SIBLING guard uses) and the control came back
# with offenders=[] -- it bounded nothing. The reason is real and worth keeping:
# at 237e5d2 the document carried no row at all for q38, 3090-inkling or
# 3090-defiant, so NO row said "SERVING" on either local host and the
# contradiction my predicate looks for did not yet exist. The defect was
# introduced by c580c80, the commit that added those three rows while leaving
# the two ollama rows saying "idle". A negative control pinned to the wrong
# pre-fix commit passes for the worst reason: the defect is absent.
PREFIX_SHA = "c580c80c3e1f06a9199c8eb470dc02a27ea2f04b"

# The lanes whose row claimed idle capacity while their host served a model.
MISLEADING_BEFORE_PORTS = ("192.168.1.232:11434", "192.168.1.124:11434")

# A qualifying phrase. Any of these makes the row honest, because each one tells
# the reader that DAEMON idleness is not MACHINE capacity.
QUALIFIERS = ("host serving", "capacity not established", "not counted as free")


def _rows(text):
    """Availability rows: a pipe-delimited line whose LABEL names a port.

    Same shape predicate as test_routing_lane_table_matches_roster.py, and for
    the same measured reason: keying on a latency figure broke the moment the
    table stopped carrying hard-coded milliseconds, and keying on a substring
    over the whole file let the prose about a lane stand in for a row for it.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if not re.search(r":\d{2,5}", cells[0]):
            continue
        out.append((cells[0], cells[1], cells[2], s))
    return out


def _host_of_label(label):
    """The MACHINE a row's label refers to, using the roster's own tunnel map.

    Reused rather than reimplemented: fleet_probe.TUNNEL_HOSTS is the one place
    that knows 127.0.0.1:8090 and 127.0.0.1:18082 are the 3090 behind an ssh
    tunnel. A row's host is not its URL's host, and hard-coding the map here
    would make the guard disagree with the probe the moment a tunnel moves.
    """
    m = re.search(r"(\d+\.\d+\.\d+\.\d+|127\.0\.0\.1):(\d{2,5})", label)
    if not m:
        return None
    url = "http://%s:%s" % (m.group(1), m.group(2))
    if url in fleet_probe.TUNNEL_HOSTS:
        return fleet_probe.TUNNEL_HOSTS[url]
    return m.group(1)


def _states(text):
    """(label, host, state_cell) for every row that names a host."""
    out = []
    for label, state, _models, _raw in _rows(text):
        host = _host_of_label(label)
        if host:
            out.append((label, host, state))
    return out


def _offenders(text):
    """Rows that claim idle capacity on a host another row says is serving.

    Returns [(label, host, state)]. Empty is the good state.
    """
    states = _states(text)
    serving_hosts = {
        host for _label, host, state in states
        if "serving" in state.lower() and "not serving" not in state.lower()
    }
    bad = []
    for label, host, state in states:
        low = state.lower()
        if "idle" not in low:
            continue
        if host not in serving_hosts:
            continue          # nothing contradicts this row on this host
        if any(q in low for q in QUALIFIERS):
            continue          # honest: it says the host is serving
        bad.append((label, host, state))
    return bad


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def test_no_row_claims_idle_capacity_on_a_host_that_is_serving():
    """MEASURED DEFECT: both local hosts served while their ollama rows said idle."""
    bad = _offenders(ROUTING.read_text(encoding="utf-8"))
    assert not bad, (
        "row(s) claim idle capacity on a host that another row says is SERVING, "
        "with no qualifier telling the reader that daemon idleness is not "
        "machine capacity: %r. Either the host is not serving (fix the other "
        "row) or this row must state 'host SERVING ... capacity NOT established'."
        % (bad,))


def test_every_local_host_row_pair_is_actually_inspected():
    """The predicate must be able to see the rows it polices.

    A guard that finds no rows would pass this file's subject test vacuously.
    fleet_probe.LANES names six lanes; the table must carry a host-bearing row
    for each of the two local machines at minimum.
    """
    states = _states(ROUTING.read_text(encoding="utf-8"))
    hosts = {h for _l, h, _s in states}
    assert "192.168.1.232" in hosts, (
        "no row resolves to the 5090's host, so this predicate inspected "
        "nothing on the machine it was written for: %r" % (hosts,))
    assert "192.168.1.124" in hosts, (
        "no row resolves to the 3090's host, so this predicate inspected "
        "nothing on the machine it was written for: %r" % (hosts,))
    assert len(states) >= 5, len(states)


def test_tunnel_rows_resolve_to_their_far_end():
    """The map that makes 'one host = one GPU' true must be the one in use.

    Without it the 3090 appears three times -- idle on its ollama daemon,
    serving Inkling, serving defiant-vision -- and the idle third invites a
    dispatch onto a card that then fails to load.
    """
    assert _host_of_label("3090-inkling `127.0.0.1:18082`") == "192.168.1.124"
    assert _host_of_label("3090-defiant `127.0.0.1:8090`") == "192.168.1.124"
    assert _host_of_label("5090 ollama `192.168.1.232:11434`") == "192.168.1.232"


# ---------------------------------------------------------------------------
# Negative control: the pre-fix document, read from a NAMED commit.
# ---------------------------------------------------------------------------

def _prefix_doc():
    """The pre-fix ROUTING.md, materialised outside this repo."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="routing-idle-prefix-"))
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         "%s:comms-lane/ROUTING.md" % PREFIX_SHA],
        capture_output=True, text=True, check=True)
    assert out.stdout, "ROUTING.md not in %s" % PREFIX_SHA
    p = d / "ROUTING.md"
    p.write_text(out.stdout, encoding="utf-8", newline="")
    return p


def test_negative_control_prefix_doc_fails_the_guard():
    """Red on the pre-fix doc by CONTENT, green here -- the arm is load-bearing.

    Asserting on the offending PORTS rather than on a count keeps this honest if
    a later edit adds rows: the control says the defect it was written for is
    present, not that the document has a particular size.
    """
    pre_bad = _offenders(_prefix_doc().read_text(encoding="utf-8"))
    pre_ports = [
        re.search(r"(\d+\.\d+\.\d+\.\d+:\d{2,5})", label).group(1)
        for label, _h, _s in pre_bad
        if re.search(r"(\d+\.\d+\.\d+\.\d+:\d{2,5})", label)
    ]
    missing = [p for p in MISLEADING_BEFORE_PORTS if p not in pre_ports]
    assert not missing, (
        "the pre-fix document at %s did not reproduce the defect for port(s) "
        "%s, so this control bounds nothing: measured offenders=%r"
        % (PREFIX_SHA, missing, pre_ports))

    assert _offenders(ROUTING.read_text(encoding="utf-8")) == [], (
        "this tree does not carry the fix, so the pin is not comparing a "
        "pre-fix document to a fixed one")
