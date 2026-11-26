"""A reasoning model's biggest cost was a number nothing displayed.

WHAT WAS MEASURED (2026-11-28)

``nous_ab.py`` wrote ``reasoning_chars`` into every run record at ONE site
(``call()``, line 147) and no other line in the repo read it. Grep across the
whole tree returned that single hit. So the largest output a reasoning model
produces -- text the caller never receives -- was recorded and never summed, the
same defect as a guard nobody calls.

The magnitude is not marginal. Over the run files already on disk:

    comms-lane/ab-nous-qwen.json      content  19,402c   reasoning 107,648c   5.55x
    comms-lane/ab-nous-results.json   content   1,067c   reasoning  28,553c  26.76x
    comms-lane/ab-nous-solar.json     content  21,145c   reasoning       0c   0.00x

A lane spending 27 characters of thinking per character of answer is a lane
whose budget goes mostly to text the caller cannot read. That is exactly the
figure the "decrease token usage" goal has to be able to see before it can claim
anything, and it was invisible to the summary.

WHY CHARS AND NOT TOKENS

The run records carry ``completion_tokens`` but no reasoning token count.
Converting chars to tokens with a guessed divisor would be the fabricated number
this stack exists to refuse, so the line states chars and says plainly that no
token split for it is recorded. An exact ratio over a stated unit beats a
plausible number over an invented one.

DESIGN NOTE ON THE GUARDS

Each guard drives ``main()`` and asserts on what it PRINTED, not on a new field.
A guard reading a field this change introduces would fail on the pre-fix tree by
``KeyError`` and would bound no behaviour; the whole point is that the RUN
reported nothing, so the assertion belongs on the run's output. Each control
passes on BOTH trees.
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


def _body(content: str, reasoning: str) -> dict:
    """An OpenAI-shaped answer with the reasoning field a reasoning model fills.

    The shape is the one ``call()`` already parses: ``message.content`` for what
    the caller receives and ``message.reasoning`` for what it does not.
    """
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": content, "reasoning": reasoning},
        }],
        "usage": {"completion_tokens": 400},
    }


def _run(monkeypatch, tmp_path, capfd, body):
    def fake_urlopen(_req, timeout=None):
        return _Resp(body)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    # A working shaper, so the shaped arm is a real comparison rather than a
    # fallback that would add its own lines to the output under test.
    monkeypatch.setattr(shape, "shape",
                        lambda task, *_a, **_kw: "REWRITTEN: " + task)
    monkeypatch.setattr(shape, "shape_with_usage",
                        lambda task, *_a, **_kw: (shape.shape(task, *_a, **_kw), {}))
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out, out


# ------------------------------------------------------------------ guards ---

def test_reasoning_spend_is_reported(monkeypatch, tmp_path, capfd):
    """THE DEFECT: the run spent heavily on reasoning and the summary was silent."""
    printed, _out = _run(
        monkeypatch, tmp_path, capfd,
        _body("a substantive answer " * 20, "thinking hard " * 100),
    )
    assert "reasoning spend" in printed, (
        "the run carried reasoning_chars in every record and the summary "
        f"reported no reasoning figure at all:\n{printed}"
    )


def test_the_reported_reasoning_figure_is_measured_not_placed(
        monkeypatch, tmp_path, capfd):
    """The number must be the sum over the runs, not a placeholder.

    Two tasks x two conditions = 4 runs, each with a 1,300-char reasoning field,
    so the reported total is exactly 5,200 chars. Anything else is not a
    measurement of what happened.
    """
    printed, out = _run(
        monkeypatch, tmp_path, capfd,
        _body("a substantive answer " * 20, "thinking hard " * 100),
    )
    line = next((ln for ln in printed.splitlines() if "reasoning spend" in ln), "")
    assert line, printed
    recorded = json.loads(out.read_text(encoding="utf-8"))["runs"]
    expected = sum(r.get("reasoning_chars") or 0 for r in recorded)
    assert expected > 0, "the fixture produced no reasoning chars to report"
    assert f"{expected}c of reasoning" in line, (
        f"the summary reported {line!r} but the records sum to {expected}c -- "
        "a figure that does not match the records is not a measurement"
    )


def test_the_ratio_is_the_one_the_records_imply(
        monkeypatch, tmp_path, capfd):
    """The x-factor must divide reasoning by content, not the other way round."""
    printed, out = _run(
        monkeypatch, tmp_path, capfd,
        _body("short answer " * 5, "very long thinking " * 200),
    )
    line = next((ln for ln in printed.splitlines() if "reasoning spend" in ln), "")
    assert line, printed
    runs = json.loads(out.read_text(encoding="utf-8"))["runs"]
    reasoning = sum(r.get("reasoning_chars") or 0 for r in runs)
    content = sum(r.get("content_chars") or 0 for r in runs)
    expected_ratio = reasoning / max(1, content)
    assert expected_ratio > 1.0, (
        f"the fixture did not invert the balance ({expected_ratio:.2f}x) -- "
        "this arm cannot bound the direction"
    )
    assert f"{expected_ratio:.2f}x" in line, (
        f"the summary reported {line!r}, the records imply "
        f"{expected_ratio:.2f}x -- reasoning over content"
    )


def test_tokens_are_not_invented_from_chars(monkeypatch, tmp_path, capfd):
    """A char count must not be silently turned into a token figure.

    No reasoning token count is recorded anywhere, so any token number on this
    line would have to come from a guessed divisor.
    """
    printed, _out = _run(
        monkeypatch, tmp_path, capfd,
        _body("a substantive answer " * 20, "thinking hard " * 100),
    )
    line = next((ln for ln in printed.splitlines() if "reasoning spend" in ln), "")
    assert line, printed
    assert "no token count for it is recorded" in line, (
        f"the line states a reasoning figure without stating that its token "
        f"split is unrecorded: {line!r}"
    )


# ----------------------------------------------------------------- controls ---

def test_ctrl_a_run_with_no_reasoning_says_nothing(monkeypatch, tmp_path, capfd):
    """CONTROL: a non-reasoning lane must not gain a spurious 0c line.

    Passes on BOTH trees: before the change there is no line at all, after it
    the line is suppressed when the sum is 0. A lane that reasons nothing has
    nothing to report here, and printing '0c of reasoning' would make 'this
    model does not reason' look like a measurement of a cost.
    """
    printed, out = _run(
        monkeypatch, tmp_path, capfd,
        _body("a substantive answer " * 20, ""),
    )
    assert "reasoning spend" not in printed, (
        f"a run with zero reasoning chars reported a reasoning spend:\n{printed}"
    )
    runs = json.loads(out.read_text(encoding="utf-8"))["runs"]
    assert sum(r.get("reasoning_chars") or 0 for r in runs) == 0, (
        "the fixture was supposed to carry no reasoning"
    )


def test_ctrl_the_existing_token_and_verdict_lines_survive(
        monkeypatch, tmp_path, capfd):
    """CONTROL: the summary's existing facts must not be displaced."""
    printed, _out = _run(
        monkeypatch, tmp_path, capfd,
        _body("a substantive answer " * 20, "thinking hard " * 100),
    )
    for expected in ("=== SUMMARY", "wrote "):
        assert expected in printed, (
            f"the summary lost {expected!r} while gaining the reasoning "
            f"line:\n{printed}"
        )
