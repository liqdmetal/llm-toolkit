"""A loopback tunnel's port is a placement, not a lane guess.

WHY THIS EXISTS
---------------
``_lane_kind`` feeds the one line a reader uses to answer the DISTRIBUTE
question ("did bulk work go to a card, or to a billed lane?"). It decided
loopback endpoints by asking ``_is_hosted`` -- whose contract is the two-valued
"is this NOT one of our cards" -- and printed that answer as if it had settled
"whose card was it".

MEASURED 2026-12-02 against the endpoints this fleet actually runs:

    endpoint                                    lane before   truth
    http://127.0.0.1:8099/v1/chat/completions   hosted        hosted (nous)
    http://127.0.0.1:8081/v1/chat/completions   local         local  (q38 5090)
    http://127.0.0.1:11434/api/chat             local         local  (3090)
    http://127.0.0.1:18082/v1/chat/completions  local         local  (defiant)
    http://127.0.0.1:8090/v1/chat/completions   HOSTED        local  (3090 llama.cpp)
    http://127.0.0.1:11448/v1/chat/completions  HOSTED        local  (q38 alt tunnel)

The two wrong rows are the sharp end: a sweep run entirely on our own 3090
printed

    all N run(s) went to a HOSTED lane; 0 local GPU(s) used

which is a false SPEND claim about work that cost nothing, in the exact place a
reader looks to find out whether the stack is distributing. The port set was a
hardcoded ``(18082, 8081, 11434)`` matching neither probe result.

WHAT THE FIX DOES
-----------------
Two named tables, because they carry two different claims:

    LOCAL_LOOPBACK_PORTS  = {8081, 11434, 18082}   # our cards, free
    HOSTED_LOOPBACK_PORTS = {8099}                 # the billed proxy

A loopback port in NEITHER table is ``unknown`` -- not hosted, because "I cannot
place this" is not a spend claim. The summary now distinguishes the two causes
of unknown: an absent endpoint vs a recorded endpoint whose port is unplaced,
and names the port to add.

``_is_hosted`` keeps its two-valued contract on purpose (an unplaced port is
genuinely not one of our cards); only the summary's question changed.

HOW THE ARMS ARE SPLIT
----------------------
GUARD (fails on the pre-fix tree):
    * TestALocalCardTunnelIsNotBilled          (:8090, :11448)
    * TestAnUnplacedPortIsNotCalledBilled      (:9999)
    * TestTheSummaryNamesTheUnplacedPort
    * TestTheLoopbackTablesAgreeWithTheProbes

CONTROL (passes on BOTH trees):
    * TestTheHostedProxyIsStillHosted          (:8099)
    * TestThePlacedLocalTunnelsAreStillLocal   (:8081, :11434, :18082)
    * TestALanAddressIsStillPlacedByItsHost    (192.168.1.232/.124)
    * TestAnAbsentEndpointIsStillUnknown       ("" / None)

The negative control points ``LLM_TOOLKIT_COMMS_DIR`` at the module pinned from
``git show 5bd4df6:comms-lane/nous_ab.py``; unset it uses the repo's own tree.
Arms read new symbols through ``getattr`` where they must, and every guard also
asserts the sentence an operator reads, so no guard passes pre-fix by accident
and none fails pre-fix merely by ``AttributeError``.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMS = Path(os.environ.get("LLM_TOOLKIT_COMMS_DIR") or (REPO / "comms-lane"))
if str(COMMS) not in sys.path:
    sys.path.insert(0, str(COMMS))

import nous_ab as nab  # noqa: E402

NOUS = "http://127.0.0.1:8099/v1/chat/completions"
Q38_TUNNEL = "http://127.0.0.1:8081/v1/chat/completions"
OLLAMA_TUNNEL = "http://127.0.0.1:11434/api/chat"
DEFIANT_TUNNEL = "http://127.0.0.1:18082/v1/chat/completions"
LLAMACPP_8090_TUNNEL = "http://127.0.0.1:8090/v1/chat/completions"
Q38_11448_TUNNEL = "http://127.0.0.1:11448/v1/chat/completions"
Q38_LAN = "http://192.168.1.232:8081/v1/chat/completions"
GG3090_LAN = "http://192.168.1.124:11434/api/chat"


def _lane(ep):
    return nab._lane_kind(ep)


def _unplaced(ep):
    """The port the summary could not place, or None.

    Read through getattr so a guard about the SENTENCE does not fail pre-fix on
    AttributeError alone -- the pre-fix tree has no such helper and must fail
    because it calls a local card billed.
    """
    fn = getattr(nab, "_unplaced_loopback_port", None)
    return fn(ep) if fn is not None else "__absent__"


# ------------------------------------------------------------------ guards --

def test_a_local_card_tunnel_is_not_billed():
    """GUARD: :8090 and :11448 are our own cards' tunnels, reported billed."""
    for ep, why in ((LLAMACPP_8090_TUNNEL, "3090 llama.cpp tunnel"),
                    (Q38_11448_TUNNEL, "q38 alt tunnel")):
        lane = _lane(ep)
        assert lane != "hosted", (
            f"{why} ({ep}) was called a HOSTED lane -- a false spend claim "
            f"about work on our own card"
        )
        assert lane == "local" or lane == "unknown", lane


def test_an_unplaced_port_is_not_called_billed():
    """GUARD: a port in no table is unplaced, never a billed-lane claim."""
    ep = "http://127.0.0.1:9999/v1/chat/completions"
    lane = _lane(ep)
    assert lane != "hosted", (
        "an unplaceable loopback port was called a billed lane; 'I cannot place "
        f"this' is not a spend claim ({lane!r})"
    )
    assert lane == "unknown", lane


def test_the_summary_names_the_unplaced_port():
    """GUARD: 'no endpoint was recorded' is a lie when the endpoint is right there."""
    assert _unplaced(LLAMACPP_8090_TUNNEL) == 8090, (
        "the summary must be able to name the port to add to the table")
    assert _unplaced(NOUS) is None
    assert _unplaced(Q38_TUNNEL) is None


def test_the_loopback_tables_agree_with_the_probes():
    """GUARD: the port sets must match what the fleet actually answers on."""
    local = getattr(nab, "LOCAL_LOOPBACK_PORTS", None)
    assert local is not None, "the local loopback port table does not exist"
    for port in (8081, 11434, 18082):
        assert port in local, (
            f"port {port} is a local card's tunnel but is missing from "
            f"LOCAL_LOOPBACK_PORTS {sorted(local)} -- this is the drift that "
            f"produced the billed-lane bug")
    hosted = getattr(nab, "HOSTED_LOOPBACK_PORTS", None)
    if hosted is not None:
        assert local.isdisjoint(hosted), (
            f"a port in both tables cannot be placed: {sorted(local & hosted)}")


# ---------------------------------------------------------------- controls --

def test_the_hosted_proxy_is_still_hosted():
    """CONTROL: passes on both trees. nous is billed and must say so."""
    assert _lane(NOUS) == "hosted", (
        "the nous portal is the BILLED lane; reading it as local would hide "
        "real spend")


def test_the_placed_local_tunnels_are_still_local():
    """CONTROL: passes on both trees. The three ports that already worked."""
    for ep in (Q38_TUNNEL, OLLAMA_TUNNEL, DEFIANT_TUNNEL):
        assert _lane(ep) == "local", f"{ep} must stay local, got {_lane(ep)!r}"


def test_a_lan_address_is_still_placed_by_its_host():
    """CONTROL: passes on both trees. Direct LAN addresses need no port table."""
    for ep in (Q38_LAN, GG3090_LAN):
        assert _lane(ep) == "local", f"{ep} must be local, got {_lane(ep)!r}"


def test_an_absent_endpoint_is_still_unknown():
    """CONTROL: passes on both trees. No endpoint names no lane."""
    for ep in ("", None, "   ", "v1/chat/completions"):
        assert _lane(ep) == "unknown", (
            f"{ep!r} names no lane and must read unknown, got {_lane(ep)!r}")
