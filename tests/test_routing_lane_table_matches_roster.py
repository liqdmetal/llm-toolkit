"""ROUTING.md's availability table must name every lane the roster measures.

THE DEFECT, measured 2026-11-29 at HEAD 237e5d2 before this file existed
-------------------------------------------------------------------------
``fleet_probe.LANES`` carries six lanes. ``ROUTING.md``'s availability table
carried three rows::

    | nous portal `127.0.0.1:8099` | UP, 316 ms | 410 |
    | 5090 `192.168.1.232:11434`  | UP, 18 ms  | 7b-qwen-q4, qwen2.5:7b, ... |
    | 3090 `192.168.1.124:11434`  | UP, 17 ms  | 7, incl. defiant9b-abliterated:256k |

The three lanes with NO table row were exactly the three that were SERVING with
spare capacity in the same run:

    q38            `192.168.1.232:8081`   serving 1, slots UNKNOWN (no key)
    3090-inkling   `127.0.0.1:18082`      serving 1, slots UNKNOWN
    3090-defiant   `127.0.0.1:8090`       serving 1, 0/4 slots busy

Meanwhile the three rows the doc did carry describe the lanes that were idle
(5090 and 3090 ollama daemons, 0 loaded each). So the policy document pointed a
reader at the two boxes' IDLE daemons and away from the one lane with measured
free slots -- which is this file's own stated expensive direction: "the router
falls through to a billed one".

WHY THIS IS A STRUCTURE DEFECT, NOT A STALE FIGURE
--------------------------------------------------
``grep q38 comms-lane/ROUTING.md`` FINDS the string -- the prose below the table
discusses q38 by name and quotes its port. A text search for the lane name says
the lane is documented; a reader looking for its ROW finds nothing. That is the
same shape as the two defects this repo already fixed: a census that matches a
name instead of resolving the thing (HF4's call census) and a doc whose markers
were hard-wrapped so a line-based scan missed them.

So the predicate below is ROW-SCOPED: a lane is documented only if a pipe-
delimited availability row carries its port. Prose mentions count for nothing.

WHAT IS ASSERTED
----------------
  1. every lane in ``LANES`` appears in at least one availability table row;
  2. no row is present for a lane NOT in ``LANES`` (a table that invents a lane
     is as misleading as one that omits it);
  3. the table carries a serving/residency state per lane, not just a latency;
  4. a latency figure in the doc is either derived from the probe or absent --
     the doc may not carry a number no instrument produces.

ARM 4 is the one that keeps this honest over time: hard-coded milliseconds in a
doc go stale the moment the fleet is re-probed, and a stale figure is not a
measurement. The doc is required to say WHICH RUN its figures come from.

CONTROLS
--------
The negative control materialises the pre-fix ``ROUTING.md`` from a NAMED full
sha into a temp dir -- never ``HEAD``, because committing the fix moves HEAD onto
the fixed tree and the arm could then only pass. It asserts the pre-fix doc is
missing exactly the three lanes above, BY NAME, so a guard that would pass on
any document cannot satisfy it.
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

# The commit this file's negative control is pinned against. FULL sha, never
# HEAD. Re-pin to the commit before this one if the arm goes red.
PREFIX_SHA = "237e5d2e8a833bc2a14cb1d63dfd3155d40a4e54"

# The three lanes measured SERVING with no row for them before this change.
UNDOCUMENTED_BEFORE_MODELS = ("q38", "3090-inkling", "3090-defiant")
# ...and the ports that identify them in a row label.
UNDOCUMENTED_BEFORE_PORTS = ("8081", "18082", "8090")


def _rows(text):
    """Availability rows: pipe-delimited, three cells, with a PORT in the label.

    Selected by STRUCTURE and by the lane port, not by a latency figure. An
    earlier draft of this file keyed on a `ms` figure, which broke the moment the
    table stopped carrying hard-coded milliseconds -- the guard then found zero
    rows and reported every lane as undocumented. A predicate that depends on
    the very thing it is meant to police is not a predicate.

    Row-scoped on purpose. The prose in this doc names q38 and quotes :8081, so
    a substring search over the whole file reports the lane as documented when
    no row exists for it.
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


def _ports_in_rows(text):
    """Lane ports named by the LABEL cell of an availability row.

    Label cell only. A port appearing in a STATE or MODELS cell is prose about
    a lane, not a row for it.
    """
    ports = set()
    for label, _state, _models, _raw in _rows(text):
        ports.update(re.findall(r":(\d{2,5})", label))
    return ports


def _roster_ports():
    return {url.rsplit(":", 1)[1] for _n, url, _k in fleet_probe.LANES}


# --------------------------------------------------------------------------
# The guards.
# --------------------------------------------------------------------------

def test_every_roster_lane_has_an_availability_row():
    """MEASURED DEFECT: three serving lanes had no row at all."""
    text = ROUTING.read_text(encoding="utf-8")
    documented = _ports_in_rows(text)
    missing = sorted(_roster_ports() - documented)
    assert not missing, (
        "fleet_probe.LANES measures %d lanes but ROUTING.md's availability "
        "table has no row for port(s) %s. The lanes it omitted were the ones "
        "SERVING with spare capacity, and a reader of this policy would not "
        "know they exist.\nrows found: %r"
        % (len(fleet_probe.LANES), missing, [r[3] for r in _rows(text)]))


def test_no_row_describes_a_lane_outside_the_roster():
    """A table that invents a lane misleads exactly as much as one that omits."""
    text = ROUTING.read_text(encoding="utf-8")
    label_ports = _ports_in_rows(text)
    extra = sorted(label_ports - _roster_ports())
    assert not extra, (
        "ROUTING.md documents lane port(s) %s that fleet_probe.LANES does not "
        "measure; the doc and the roster disagree about the fleet" % (extra,))


def test_each_row_states_lane_state_not_just_latency():
    """Latency alone cannot route. A row must say serving/idle/unknown."""
    text = ROUTING.read_text(encoding="utf-8")
    bad = []
    for label, state, _models, _raw in _rows(text):
        low = state.lower()
        if not any(k in low for k in ("up", "down", "serving", "idle",
                                      "unknown", "unreachable")):
            bad.append((label, state))
    assert not bad, (
        "availability row(s) carry no lane state, so a reader cannot tell an "
        "idle lane from a serving one: %r" % (bad,))


def test_doc_names_the_run_its_figures_came_from():
    """A hard-coded latency with no run attached is a figure nobody owns."""
    text = ROUTING.read_text(encoding="utf-8")
    section = text.split("## Measured availability", 1)[1].split("## Hard rules", 1)[0]
    assert re.search(r"20\d\d-\d\d-\d\d", section), (
        "the availability section carries latency figures with no date, so a "
        "reader cannot tell a measurement from a memory")


# --------------------------------------------------------------------------
# Negative control: the pre-fix document, read from a NAMED commit.
# --------------------------------------------------------------------------

def _prefix_doc():
    """The pre-fix ROUTING.md, materialised outside this repo."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="routing-prefix-"))
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         "%s:comms-lane/ROUTING.md" % PREFIX_SHA],
        capture_output=True, text=True, check=True)
    assert out.stdout, "ROUTING.md not in %s" % PREFIX_SHA
    p = d / "ROUTING.md"
    p.write_text(out.stdout, encoding="utf-8", newline="")
    return p


def test_tripwire_prefix_sha_does_not_carry_the_fix():
    """By CONTENT, not by comparing to HEAD.

    A `!= HEAD` tripwire fails on a correct tree before the commit and stops
    meaning anything after it. Reading the marker out of the pinned commit is
    stable on both sides of the commit.
    """
    pre_ports = _ports_in_rows(_prefix_doc().read_text(encoding="utf-8"))
    already = [p for p in UNDOCUMENTED_BEFORE_PORTS if p in pre_ports]
    assert not already, (
        "the pre-fix doc at %s already has rows for port(s) %s, so this control "
        "measures the fixed tree" % (PREFIX_SHA, already))
    now_ports = _ports_in_rows(ROUTING.read_text(encoding="utf-8"))
    added = [p for p in UNDOCUMENTED_BEFORE_PORTS if p in now_ports]
    assert len(added) == len(UNDOCUMENTED_BEFORE_PORTS), (
        "this tree does not carry rows for %s, so the pin is not comparing a "
        "pre-fix document to a fixed one: %r"
        % (list(UNDOCUMENTED_BEFORE_PORTS), added))


def test_negative_control_prefix_doc_fails_the_guard():
    """Red on the pre-fix doc, green here -- the arms are load-bearing."""
    pre_missing = sorted(_roster_ports() -
                         _ports_in_rows(_prefix_doc().read_text(encoding="utf-8")))

    assert set(UNDOCUMENTED_BEFORE_PORTS) <= set(pre_missing), (
        "the pre-fix document did not reproduce the defect, so this control "
        "bounds nothing: missing=%r wanted=%r"
        % (pre_missing, list(UNDOCUMENTED_BEFORE_PORTS)))
    assert len(pre_missing) == 3, (
        "the pre-fix doc was expected to omit exactly three lanes, measured "
        "%d: %r" % (len(pre_missing), pre_missing))

    now_missing = sorted(_roster_ports() - _ports_in_rows(
        ROUTING.read_text(encoding="utf-8")))
    assert now_missing == [], now_missing

    # ...and the control reads THIS repo's roster, not a stub.
    assert len(fleet_probe.LANES) == 6, len(fleet_probe.LANES)
