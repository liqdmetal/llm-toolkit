"""Tokens the shaper spent and bought nothing with must still be reported.

`nous_ab.py`'s summary has one block that reports what the SHAPER spent, and it
was built from `applied` -- the shaped runs whose rewrite the harness actually
USED. But a dispatch is the unit that costs: the shaper is called, the model
generates, the endpoint bills, and only THEN does the harness decide whether the
rewrite is usable. A shaper that ran and returned nothing usable is therefore
the most expensive shape of this trade -- full bill, zero benefit -- and it
contributed NOTHING to the total.

MEASURED 2026-12-03 on this tree, stub frontier, stub shaper, both arms paying
237 prompt + 125 completion tokens per dispatch:

    arm   shaper result                 pre-fix summary printed
    A     rewrite + usage               "shaper tokens: 474 in + 250 out ..."
    B     no rewrite + usage            (nothing about tokens at all -- only
                                         "shaped arm NEVER SHAPED")
    C     unreachable, no usage         (nothing, correctly)

Arm B paid 474 prompt and 250 completion tokens and the report read as a sweep
that spent nothing to learn nothing. The failure is in the good direction for
nobody: a reader cannot see that shaping is bleeding tokens, so the one lever
the "decrease token usage" goal needs to pull is the lever the report hides.

WHAT THESE ARMS PIN

    B must print the spend it paid      (the defect)
    the figure must equal what the
      records say was paid              (measured, not a plausible number)
    the dispatch count must sit beside
      the usable count                  (so a zero-benefit sweep cannot read
                                         as a productive one)
    C must still print no figure        (a soft dependency that failed spent
                                         nothing -- control, passes both trees)
    A must be unchanged                 (no regression in the working arm)

Each guard drives ``main()`` and asserts on what it PRINTED and RECORDED, never
on a name this change introduces: a guard importing a new symbol would fail on
the pre-fix tree with ImportError and bound no behaviour. The control passes on
BOTH trees, which is what makes it a control.
"""
import io
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "comms-lane"
sys.path.insert(0, str(LANE))

import nous_ab  # noqa: E402
import shape  # noqa: E402


class _Resp(io.BytesIO):
    def __init__(self, payload: dict):
        super().__init__(json.dumps(payload).encode())
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _frontier_body() -> dict:
    return {
        "choices": [{"finish_reason": "stop",
                     "message": {"content": "a substantive answer",
                                 "reasoning": ""}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
    }


# The two numbers shape.py's own docstring measured against the live box. A
# dispatch that produced nothing usable still paid exactly these.
SPENT = {
    "model": "defiant9b-abliterated:256k",
    "basis": "measured",
    "prompt_tokens": 237,
    "completion_tokens": 125,
}


def _run(monkeypatch, tmp_path, capfd, *, arm):
    """Drive main() for one arm. Returns (printed, receipt).

    arm "applied"  the shaper answered AND its rewrite was used
    arm "spent"    the shaper answered, paid tokens, produced no usable rewrite
    arm "dead"     the shaper could not be reached at all (no usage reported)
    """
    def fake_urlopen(_req, timeout=None):
        return _Resp(_frontier_body())

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)

    def _rewrite(task, *_a, **_kw):
        return "REWRITTEN: " + task

    if arm == "applied":
        monkeypatch.setattr(shape, "shape", _rewrite)
        monkeypatch.setattr(
            shape, "shape_with_usage",
            lambda task, *_a, **_kw: (_rewrite(task),
                                      dict(SPENT, expansion_x=3.13,
                                           produced_text=True)))
    elif arm == "spent":
        # Paid for 362 tokens and handed back an empty string: the model ran,
        # hit its cap or refused, and produced nothing the harness could use.
        monkeypatch.setattr(shape, "shape", lambda *_a, **_kw: "")
        monkeypatch.setattr(
            shape, "shape_with_usage",
            lambda task, *_a, **_kw: ("",
                                      dict(SPENT, expansion_x=None,
                                           produced_text=False)))
    else:
        def _raise(*_a, **_kw):
            raise shape.ShaperUnavailable("shaper host unreachable (fixture)")
        monkeypatch.setattr(shape, "shape", _raise)
        monkeypatch.setattr(shape, "shape_with_usage", _raise)

    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out, json.loads(out.read_text(encoding="utf-8"))


def _paid(receipt):
    """What the records say the shaper was actually paid, over every dispatch."""
    shaped = [r for r in receipt["runs"] if r.get("condition") == "shaped"]
    pin = sum(r["shaper_prompt_tokens"] for r in shaped
              if isinstance(r.get("shaper_prompt_tokens"), int))
    pout = sum(r["shaper_completion_tokens"] for r in shaped
               if isinstance(r.get("shaper_completion_tokens"), int))
    return pin, pout, shaped


# ------------------------------------------------------------------ guards ---

def test_a_shaper_that_spent_and_bought_nothing_still_reports_its_spend(
        monkeypatch, tmp_path, capfd):
    """THE DEFECT: 474 prompt + 250 completion tokens paid, no figure printed."""
    printed, receipt = _run(monkeypatch, tmp_path, capfd, arm="spent")
    pin, pout, shaped = _paid(receipt)

    assert shaped, "precondition: the fixture produced shaped runs"
    assert pin == 237 * len(shaped) and pout == 125 * len(shaped), (
        "precondition: every dispatch recorded what it cost, "
        f"records hold {pin}/{pout} over {len(shaped)} dispatch(es)")
    assert not any(r.get("shaped_applied") for r in shaped), (
        "precondition: this arm must be a ZERO-BENEFIT sweep -- no rewrite "
        "was usable")

    assert "shaper tokens:" in printed, (
        "the shaper was paid "
        f"{pin} prompt and {pout} completion tokens over {len(shaped)} "
        "dispatch(es) and produced no usable rewrite, and the summary said "
        f"nothing about the spend:\n{printed}"
    )
    line = next(ln for ln in printed.splitlines()
                if "shaper tokens:" in ln)
    assert str(pin) in line and str(pout) in line, (
        f"the printed spend does not state what was paid ({pin} in / {pout} "
        f"out):\n{line}")


def test_the_spend_figure_on_a_zero_benefit_sweep_is_what_was_paid(
        monkeypatch, tmp_path, capfd):
    """The total must equal the records, not a plausible number."""
    printed, receipt = _run(monkeypatch, tmp_path, capfd, arm="spent")
    pin, pout, shaped = _paid(receipt)
    assert "shaper tokens:" in printed, printed
    line = next(ln for ln in printed.splitlines() if "shaper tokens:" in ln)
    assert str(pin) in line, (
        f"records hold {pin} prompt tokens paid and the line states another "
        f"figure:\n{line}")
    assert str(len(shaped)) in line, (
        f"{len(shaped)} dispatch(es) were paid for and the line does not say "
        f"over how many:\n{line}")


def test_the_dispatch_count_sits_beside_the_usable_count(
        monkeypatch, tmp_path, capfd):
    """A zero-benefit sweep must not read as a productive one.

    "over 2 shaping dispatch(es)" is TRUE of arm B and leaves the reader to
    assume both bought something. The number that separates the arms is how
    many produced a usable rewrite, so it has to be on the same line.
    """
    printed, receipt = _run(monkeypatch, tmp_path, capfd, arm="spent")
    pin, pout, shaped = _paid(receipt)
    assert shaped and not any(r.get("shaped_applied") for r in shaped)
    line = next((ln for ln in printed.splitlines()
                 if "shaper tokens:" in ln), "")
    assert line, f"no shaper-token line at all:\n{printed}"
    assert "0 of which" in line or "0 produced" in line, (
        "the line reports the spend but not that ZERO of those dispatches "
        f"produced a usable rewrite:\n{line}")


def test_the_working_arm_is_unchanged(monkeypatch, tmp_path, capfd):
    """No regression: a fully applied sweep still reports spend and benefit."""
    printed, receipt = _run(monkeypatch, tmp_path, capfd, arm="applied")
    pin, pout, shaped = _paid(receipt)
    applied = [r for r in shaped if r.get("shaped_applied")]
    assert applied and len(applied) == len(shaped), (
        "precondition: every shaped run applied its rewrite")
    line = next((ln for ln in printed.splitlines()
                 if "shaper tokens:" in ln), "")
    assert line, f"the working arm printed no shaper spend:\n{printed}"
    assert str(pin) in line and str(pout) in line, line
    assert f"{len(applied)} of which" in line or f"{len(applied)} produced" in line, (
        f"the line must state that {len(applied)} dispatch(es) bought a "
        f"rewrite:\n{line}")


# ---------------------------------------------------------------- controls ---

def test_ctrl_an_unreachable_shaper_invents_no_spend(
        monkeypatch, tmp_path, capfd):
    """A soft dependency that failed spent nothing and must claim nothing.

    This is the arm that stops the fix from over-reporting: widening the sums
    to every shaped run must not make an UNREACHED shaper look like a billed
    one. Passes on both trees.
    """
    printed, receipt = _run(monkeypatch, tmp_path, capfd, arm="dead")
    _pin, _pout, shaped = _paid(receipt)
    assert shaped, "precondition: the fixture produced shaped runs"
    for r in shaped:
        assert r.get("shaper_prompt_tokens") is None, (
            "the shaper was never reached and the record holds a prompt token "
            f"figure anyway: {r.get('shaper_prompt_tokens')!r}")
        assert r.get("shaper_completion_tokens") is None, (
            "the shaper was never reached and the record holds a completion "
            f"token figure anyway: {r.get('shaper_completion_tokens')!r}")
    assert "shaper tokens:" not in printed, (
        "nothing was paid to the shaper and the summary invented a spend "
        f"figure:\n{printed}")
    assert "NEVER SHAPED" in printed, (
        "the arm still has to say the two arms were the same prompt, or a "
        f"reader takes the missing token line for a clean result:\n{printed}")


def test_ctrl_a_zero_benefit_sweep_says_the_arms_are_identical_prompts(
        monkeypatch, tmp_path, capfd):
    """The token figure must not replace the warning about the comparison."""
    printed, _receipt = _run(monkeypatch, tmp_path, capfd, arm="spent")
    assert "NEVER SHAPED" in printed, (
        "every shaped run fell back to the raw text, so any raw-vs-shaped "
        "delta is the lane's drift -- and the report dropped that warning "
        f"once it started printing a spend:\n{printed}")
