"""A VRAM figure nobody measured rendered as a measured zero.

``bench_local.ollama_residency`` divides ``size_vram`` by ``size`` to say whether
a model actually reached the card. That column exists because of the specific
trap this fleet keeps falling into: a model served from system RAM and one on the
card differ by an order of magnitude in decode rate and read identically in a
chat answer. So the column is load-bearing, and it was computing an answer from
fields the server may simply not have sent.

MEASURED on the pre-fix tree by driving the shipped functions with a stubbed
``_post``, four payload shapes:

    size sent, size_vram sent      -> pct 100.0 / 0.0   (a real measurement)
    size ABSENT                    -> total 0.0, pct 0.0
    size_vram ABSENT, size sent    -> vram 0.0, pct 0.0

The last two print ``0% (0.00 GB)`` -- the row for a model the tool NEVER MEASURED
is byte-identical to the row for a model it measured sitting entirely in system
RAM. That is the repo's own defect class pointed at the one column whose whole
purpose is to refuse it: an absent field collapsed into a zero, rendered as data.

AND TWO MORE IN THE SAME FILE, measured the same way:

    decode_tok_s = 0.0 (a real measurement)  -> rendered "--"
    decode_tok_s = None (never sampled)      -> rendered "--"

``if r.get("decode_tok_s")`` is falsy for both. A rate that measured zero is a
finding -- a lane that answers but produces nothing -- and it rendered as the
same dash as a lane that was never asked.

    residency probe raised      -> gpu cell "--"
    residency answered, no rows -> gpu cell "--"
    residency answered, model not in it -> "not loaded"

A probe that FAILED and a probe that answered "nothing is loaded" have entirely
different fixes: retry the lane versus dispatch nowhere. They shared a dash.

WHY THIS IS NOT COSMETIC: the tool's docstring states its own purpose -- "a tok/s
figure with no residency is the trap this fleet keeps falling into". Rendering an
unmeasured residency as a measured CPU-resident one puts the trap back inside the
instrument built to catch it. An operator reading ``0%`` concludes the card is
free; the truth is nobody asked.

THE FIX IS THREE STATES, PER THE REPO'S STANDING RULE -- probed-and-empty,
probed-and-loaded, and never-probed are different facts. ``size``/``size_vram``
now read ``None`` when absent, ``pct_on_gpu`` is ``None`` when it cannot be
computed, and ``residency_state`` keeps "the probe failed" apart from "the
server holds nothing". A null FIELD inside a reported object stays a value the
server chose, exactly as ``ssllabs_analyze``'s ``endpoints_state`` already
distinguishes it.

NEGATIVE CONTROL: nine arms below fail on the saved pre-fix copy
(sha256 0405c517…) by value, while three controls pass on both trees -- so the
guards are not satisfied by a function that refuses everything or by one that
stamps every row "unmeasured".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import bench_local  # noqa: E402


def _residency_with(monkeypatch, payload):
    """Drive the shipped probe with a canned /api/ps body."""
    monkeypatch.setattr(bench_local, "_post",
                        lambda url, body, timeout=20: payload)
    return bench_local.ollama_residency("http://stub")


def _residency_raising(monkeypatch):
    def boom(url, body, timeout=20):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(bench_local, "_post", boom)
    return bench_local.ollama_residency("http://stub")


def _row(**kw):
    base = {"lane": "x", "model": "m"}
    base.update(kw)
    return base


# ---------------------------------------------------------------- guard arms


def test_guard_absent_size_is_not_a_zero(monkeypatch):
    """The defect. An unsent ``size`` must not become total 0.0."""
    rows = _residency_with(monkeypatch, {"models": [{"name": "m", "size_vram": 0}]})
    assert rows[0]["total_gb"] is None, (
        "size was never sent, so there is no total to report")
    assert rows[0]["size_reported"] is False


def test_guard_absent_size_yields_no_percentage(monkeypatch):
    rows = _residency_with(monkeypatch, {"models": [{"name": "m", "size_vram": 0}]})
    assert rows[0]["pct_on_gpu"] is None


def test_guard_absent_vram_is_not_a_zero(monkeypatch):
    """The mirror case: ``size`` sent, ``size_vram`` never sent."""
    rows = _residency_with(
        monkeypatch, {"models": [{"name": "m", "size": 10_000_000_000}]})
    assert rows[0]["vram_gb"] is None
    assert rows[0]["pct_on_gpu"] is None


def test_guard_unmeasured_residency_does_not_render_as_zero_percent(
        monkeypatch):
    """The rendering arm: the pre-fix string was '0% (0.00 GB)'."""
    rows = _residency_with(monkeypatch, {"models": [{"name": "m", "size_vram": 0}]})
    row = _row(decode_tok_s=11.6, residency_state="loaded", residency=rows)
    table = bench_local.render([row])
    assert "0% (0.00 GB)" not in table, (
        "an unmeasured row rendered as a measured CPU-resident one")
    assert "size not reported" in table


def test_guard_failed_probe_is_named_not_dashed(monkeypatch):
    """A failed /api/ps and an empty /api/ps had the same cell."""
    failed = bench_local.render([_row(decode_tok_s=11.6,
                                      residency_state="unprobed",
                                      residency=None)])
    empty = bench_local.render([_row(decode_tok_s=11.6,
                                     residency_state="empty",
                                     residency=[])])
    assert failed != empty, "a failed probe and an empty server rendered identically"
    assert "unmeasured" in failed
    assert "none loaded" in empty


def test_guard_measured_zero_rate_is_not_a_missing_sample():
    """0.0 tok/s is a measurement; None is the absence of one."""
    measured = bench_local.render([_row(decode_tok_s=0.0,
                                        residency_state="empty", residency=[])])
    missing = bench_local.render([_row(decode_tok_s=None,
                                       residency_state="empty", residency=[])])
    assert measured != missing, (
        "a lane that measured 0.0 tok/s rendered as a lane never sampled")
    assert "0.0" in measured


def test_guard_probe_failure_is_not_reported_as_empty(monkeypatch):
    """residency_state must tell a failed probe from an empty one."""
    assert _residency_raising(monkeypatch) is None
    monkeypatch.setattr(bench_local, "_post",
                        lambda url, body, timeout=20: {"models": []})
    assert bench_local.ollama_residency("http://stub") == []


def test_guard_reported_size_is_flagged_as_reported(monkeypatch):
    """The companion guard: a sent ``size`` must be recorded as sent.

    Split out of the control it was first written inside, because an assertion
    on a key the fix introduces fails on the pre-fix tree and therefore bounds
    nothing when placed in a control.
    """
    rows = _residency_with(monkeypatch, {"models": [
        {"name": "m", "size": 10_000_000_000, "size_vram": 0}]})
    assert rows[0]["size_reported"] is True


def test_guard_state_is_recorded_for_a_loaded_probe(monkeypatch):
    """The probe result carries its state, not just its rows."""
    monkeypatch.setattr(bench_local, "_post",
                        lambda url, body, timeout=20: {"models": [
                            {"name": "qwen2.5:7b", "size": 5_070_000_000,
                             "size_vram": 0}]})
    row = bench_local.measure(("lane", "http://stub", "qwen2.5:7b", "ollama"), 1)
    assert row["residency_state"] == "loaded"
    # The failure rate is stubbed out; what matters is the state, not the rate.
    assert row["residency"][0]["pct_on_gpu"] == 0.0


# -------------------------------------------------------------------- controls
# These pass on BOTH trees. They are what stops the guards from being satisfied
# by a reader that refuses every payload or stamps everything "unmeasured".


def test_control_on_card_still_reports_one_hundred_percent(monkeypatch):
    rows = _residency_with(monkeypatch, {"models": [
        {"name": "m", "size": 10_000_000_000, "size_vram": 10_000_000_000}]})
    assert rows[0]["pct_on_gpu"] == 100.0
    assert rows[0]["total_gb"] == pytest.approx(10.0)
    assert rows[0]["vram_gb"] == pytest.approx(10.0)


def test_control_measured_cpu_residency_still_reports_zero(monkeypatch):
    """A real 0% must keep rendering as 0%, or the fix broke the column.

    Touches NO key the fix introduces -- that is what makes it a control: it
    passes on the pre-fix tree too, so the guards cannot be satisfied by a
    reader that refuses every payload.
    """
    rows = _residency_with(monkeypatch, {"models": [
        {"name": "m", "size": 10_000_000_000, "size_vram": 0}]})
    assert rows[0]["pct_on_gpu"] == 0.0
    row = _row(decode_tok_s=11.6, residency_state="loaded", residency=rows)
    assert "0% (0.00 GB)" in bench_local.render([row])


def test_control_model_absent_from_ps_still_says_not_loaded(monkeypatch):
    rows = _residency_with(monkeypatch, {"models": [
        {"name": "other", "size": 1_000_000_000, "size_vram": 1_000_000_000}]})
    table = bench_local.render([_row(decode_tok_s=11.6,
                                     residency_state="loaded", residency=rows)])
    assert "not loaded" in table
