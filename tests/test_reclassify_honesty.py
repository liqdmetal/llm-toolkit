"""reclassify must not turn a crashed run into an answer, and must not die on one bad row.

The comms-lane stack exists to refuse unmeasured claims. This file guards the
*reporter* of the A/B harnesses for the same reason: a report that calls a crash
an answer is a measurement that lies, and a report that dies on one malformed
record silently drops every other row in the file.

MEASURED 2026-11-24 at HEAD ea80b99, over the three saved result files.

DEFECT 1 -- a crashed run whose preview carries text reads as an ANSWER.
    comms-lane/ab-results.json holds one run: status ERROR, exit 127,
    out_chars 0, head ''. The OLD classify() returned ERROR there only by luck,
    because the head happened to be empty. Given a 356-character head of error
    text -- exactly what a spawn failure prints -- the old code returned
    **ANSWER**, because classify() reads `head` and nothing else. The run record
    carries `status` and `exit`; the classifier never looked at either.

DEFECT 2 -- one row without a `condition` crashes the whole file.
    `print(f"{r.get('condition'):6s}")` raises
    `TypeError: unsupported format string passed to NoneType.__format__`.
    Measured on ab-results.json itself: the tool printed the file header, then
    raised, so the run that DID fail was the one never reported.

The fix adds classify_run(run) -> (verdict, why) with a stated precedence, and
tolerant formatting in the CLI. classify(text) is unchanged, so the A/B harness
keeps its own entry point.

CONTROLS
    A pin saying "crashes are errors now" would pass a version that reported
    ERROR for everything, so this file also proves a healthy run still reads as
    ANSWER/PARTIAL, that the exit code alone is enough to fail a run, and that a
    complete report lists every row (the old code's failure mode was losing
    rows, not mislabelling one).

    python -m pytest tests/test_reclassify_honesty.py -q
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "comms-lane"
PY = sys.executable

if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import reclassify as R  # noqa: E402

# A refusal-shaped, substantive preview: what a spawn failure or a guarded model
# both print. 356 chars so it clears classify()'s >200 "substantive" bar.
ERROR_TEXT = ("Could not find codebuff. Command exited with code 127 while "
              "spawning the worker process; no request was sent. ") * 4


def ok_run(head="Here is the full analysis. " * 20, **kw):
    base = {"name": "framing", "condition": "raw", "status": "ANSWER",
            "finish": "stop", "head": head, "content_chars": len(head)}
    base.update(kw)
    return base


class CrashIsNotAnAnswerTests(unittest.TestCase):
    """Defect 1: run metadata outranks a text preview."""

    def test_an_error_status_beats_a_substantive_looking_head(self):
        """THE DEFECT. Old code returned ANSWER for this exact record shape.

        A run that never executed cannot have answered, whatever its 300-char
        preview happens to contain.
        """
        run = {"name": "smoke", "status": "ERROR", "exit": 127,
               "out_chars": 0, "head": ERROR_TEXT}
        verdict, why = R.classify_run(run)
        self.assertEqual(verdict, "ERROR")
        self.assertIn("did not complete", why)
        self.assertIn("127", why)

    def test_control_the_same_text_without_a_failed_status_is_classified_by_text(self):
        """Without this, a version that returned ERROR unconditionally would pass.

        Identical head text, healthy status: the text rule must decide, and it
        must be able to reach a non-ERROR verdict.
        """
        run = ok_run(head="A plain substantive answer. " * 20)
        verdict, _ = R.classify_run(run)
        self.assertEqual(verdict, "ANSWER")

    def test_a_non_zero_exit_alone_fails_a_run_with_no_status_field(self):
        """The exit code is evidence even when status was never recorded."""
        run = ok_run(exit=1)
        run.pop("status")
        verdict, why = R.classify_run(run)
        self.assertEqual(verdict, "ERROR")
        self.assertIn("exit=1", why)

    def test_control_a_zero_exit_does_not_fail_a_run(self):
        run = ok_run(exit=0)
        self.assertEqual(R.classify_run(run)[0], "ANSWER")

    def test_the_real_failing_record_from_disk_is_reported_as_an_error(self):
        """The actual saved run, not a fixture: ab-results.json's smoke entry."""
        data = json.loads((LANE / "ab-results.json").read_text(encoding="utf-8"))
        run = data["runs"][0]
        self.assertEqual(run["status"], "ERROR")
        self.assertEqual(run["exit"], 127)
        verdict, why = R.classify_run(run)
        self.assertEqual(verdict, "ERROR")
        self.assertIn("did not complete", why)


class HealthyRunsStillClassifyTests(unittest.TestCase):
    """The text rule is untouched; pollution of the verdicts would be worse."""

    def test_a_refusal_with_substance_is_partial(self):
        run = ok_run(head="I can't help with that. " + "But here is context. " * 15)
        self.assertEqual(R.classify_run(run)[0], "PARTIAL")

    def test_a_short_refusal_is_a_refusal(self):
        run = ok_run(head="I cannot help with this.")
        self.assertEqual(R.classify_run(run)[0], "REFUSE")

    def test_empty_text_with_no_failed_status_is_still_an_error(self):
        run = ok_run(head="")
        self.assertEqual(R.classify_run(run)[0], "ERROR")

    def test_classify_is_unchanged_for_string_callers(self):
        """The A/B harness passes strings; its entry point must keep working."""
        self.assertEqual(R.classify("I cannot help with this."), "REFUSE")
        self.assertEqual(R.classify(""), "ERROR")
        self.assertEqual(R.classify("plain text"), "ANSWER")


class TheCliReportsEveryRowTests(unittest.TestCase):
    """Defect 2: one malformed record must not cost the rest of the file."""

    def test_a_run_without_a_condition_no_longer_crashes_the_report(self):
        """THE DEFECT. Old CLI: TypeError, whole file lost after the header.

        Asserted by running the real script as a subprocess, because the crash
        was in the format expression, not in any function a unit test reaches.
        """
        proc = subprocess.run(
            [PY, str(LANE / "reclassify.py"), str(LANE / "ab-results.json")],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0,
                         f"reclassify exited {proc.returncode}: {proc.stderr[-500:]}")
        self.assertNotIn("TypeError", proc.stderr)
        self.assertIn("smoke", proc.stdout)
        self.assertIn("ERROR", proc.stdout)

    def test_control_the_report_still_lists_every_run_in_a_healthy_file(self):
        """The fix must not buy stability by dropping rows."""
        proc = subprocess.run(
            [PY, str(LANE / "reclassify.py"), str(LANE / "ab-nous-solar.json")],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        data = json.loads((LANE / "ab-nous-solar.json").read_text(encoding="utf-8"))
        for run in data["runs"]:
            with self.subTest(run=run["name"], condition=run["condition"]):
                self.assertIn(run["name"], proc.stdout)

    def test_control_all_three_files_report_in_one_invocation(self):
        """The old code died on the first bad file; the rest never printed."""
        proc = subprocess.run(
            [PY, str(LANE / "reclassify.py"),
             str(LANE / "ab-results.json"),
             str(LANE / "ab-nous-qwen.json"),
             str(LANE / "ab-nous-solar.json")],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        for name in ("ab-results", "ab-nous-qwen", "ab-nous-solar"):
            with self.subTest(file=name):
                self.assertIn(name, proc.stdout)


class ThePreviewLimitIsStatedTests(unittest.TestCase):
    """`head` is 300 chars; a reclassify verdict is therefore a preview verdict.

    Measured: ab-nous-qwen.json framing/raw is saved as ANSWER (classified by
    nous_ab.py from the FULL 4,682-char content) but reclassify calls it PARTIAL,
    because the first 300 chars are a refusal. That is a real, permanent limit of
    reclassifying from a preview, not a bug to paper over -- so it is pinned, and
    the tool must not present its verdict as the saved one.
    """

    def test_a_preview_verdict_may_differ_from_the_saved_full_text_verdict(self):
        data = json.loads((LANE / "ab-nous-qwen.json").read_text(encoding="utf-8"))
        differing = [
            r for r in data["runs"]
            if r.get("status") and R.classify_run(r)[0] != r["status"]
        ]
        self.assertTrue(differing,
                        "expected the known preview-vs-full-text difference; if "
                        "it is gone, this pin and the docstring are stale")
        for run in differing:
            with self.subTest(run=run["name"], condition=run["condition"]):
                self.assertGreater(run.get("content_chars", 0), 300,
                                   "a difference only arises past the preview cut")

    def test_control_the_record_carries_the_field_that_explains_it(self):
        """content_chars is what makes the discrepancy diagnosable."""
        data = json.loads((LANE / "ab-nous-qwen.json").read_text(encoding="utf-8"))
        for run in data["runs"]:
            with self.subTest(run=run["name"]):
                self.assertIn("content_chars", run)


if __name__ == "__main__":
    unittest.main()
