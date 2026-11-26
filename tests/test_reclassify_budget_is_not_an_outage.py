"""A spent budget is not an outage, and a stored status beats a text preview.

MEASURED on this tree 2026-12-02, driving `classify_run()` over the four saved
result files and over synthetic records.

DEFECT 1 -- reclassify re-made the collapse its own sibling writer had just fixed.
    comms-lane/nous_ab.py:191 sets `status = "TRUNCATED" if finish == "length"
    else "EMPTY"` and states outright why:

        "Without this line a 4/4-truncated sweep and a 4/4-lane outage print the
         same shape, and the reader is pushed to the wrong lever."

    `reclassify._FAILED_STATUSES` was {"ERROR", "CRASH", "TIMEOUT"}: neither word
    was present. On ab-nous-results.json (model z-ai/glm-5.3-flashx) the CLI
    printed

        framing  raw    -> ERROR    (classified from saved answer text)
        giant    raw    -> ERROR    (classified from saved answer text)

    for runs whose own `status` field says TRUNCATED. A reader is told the lane
    failed when the budget ran out -- the very misdirection nous_ab.py names.

    And it read ERROR by LUCK: those records carry `head: ""`, and classify("")
    is ERROR. Give the same status a substantive head -- what a partially
    answered, then cut-off run looks like -- and the verdict was ANSWER, a
    finished answer:

        status=TRUNCATED  head=''         -> ERROR
        status=TRUNCATED  head=400 chars  -> ANSWER

    So the verdict rested on a 300-char PREVIEW, which is the exact precedence
    classify_run() was written to stop, one field over.

DEFECT 2 -- the exit code was read by accident of Python's type lattice.
    `isinstance(exit_code, int) and exit_code != 0` is too narrow and too wide at
    once. MEASURED:

        exit='127'  (a shell wrote it)  -> ANSWER   a crash read as an answer
        exit=1.0    (float convention)  -> ANSWER   same
        exit=True                       -> ERROR "exit=True"   (bool IS an int)
        exit=False                      -> ANSWER                (False == 0)

    All twelve successful runs on disk carry `exit: None`, so none of these bit
    today; the string and float shapes are what a shell or a JSON round-trip
    produces, and the bool shapes were being read without any stated rule.

THE FIX
    `_NON_ANSWER_STATUSES = {"TRUNCATED", "EMPTY"}` leaves the ANSWER bucket but
    does NOT become ERROR: both return "NO ANSWER" with the lever named -- raise
    AB_MAX_TOKENS for TRUNCATED, "had room and wrote nothing" for EMPTY. The head
    is irrelevant to that verdict, so the preview can no longer flip it.
    The exit field is parsed by stated rule: int / float / numeric string, bool
    treated as a flag (True fails, conservatively, and says "flag set" rather
    than printing `exit=True`), and a non-numeric string is carried into the
    "why" as UNREADABLE so an unreadable exit is never mistaken for a clean one.

CONTROLS are green on BOTH trees: a completed run with a good head still reads
ANSWER, a refusal still reads REFUSE/PARTIAL, a zero exit still passes, and an
ERROR status still outranks a substantive head. Those bound the fix from the
"report NO ANSWER for everything" side, which would pass every guard here.
"""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "comms-lane"
PY = sys.executable

LONG = ("Here is the full analysis. " * 20).strip()

# A run that completed with a substantive, non-refusing answer.
def ok_run(**kw):
    base = {"name": "framing", "condition": "raw", "status": "ANSWER",
            "finish": "stop", "exit": None, "head": LONG}
    base.update(kw)
    return base


def load(path=None):
    """Load reclassify from `path`.

    Default: this tree's copy. `RECLASSIFY_MODULE_UNDER_TEST` overrides it, which
    is what the negative control uses -- set it to a `git show HEAD:` copy and the
    GUARDS below must go red while the CONTROLS stay green.

    My first draft put that override on three separate arms instead, so with the
    pre-fix module pinned the whole file still passed 19/19: the ten real guards
    kept loading the FIXED tree and the run proved nothing about the defect. A
    control that cannot reach the subject is not a control.
    """
    import os
    path = path or os.environ.get("RECLASSIFY_MODULE_UNDER_TEST")
    path = LANE / "reclassify.py" if not path else Path(path)
    spec = importlib.util.spec_from_file_location(
        "reclassify_under_test_%d" % abs(hash(str(path))), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load()


class SpentBudgetIsNotAnOutage(unittest.TestCase):
    """GUARDS: each fails on the pre-fix module, naming the collapse."""

    def test_a_truncated_run_is_not_reported_as_a_completed_answer(self):
        """The collapse: cut-off text read as a finished answer."""
        verdict, why = R.classify_run({"name": "giant", "status": "TRUNCATED",
                                       "exit": None, "head": LONG})
        self.assertNotEqual(
            verdict, "ANSWER",
            "a run whose own status says TRUNCATED is reported as ANSWER, so a "
            "spent budget reads as a finished answer (why was %r)" % (why,))

    def test_a_truncated_run_is_not_reported_as_an_outage(self):
        """The other half: ERROR sends the reader to check the lane, not the cap."""
        verdict, why = R.classify_run({"name": "giant", "status": "TRUNCATED",
                                       "exit": None, "head": LONG})
        self.assertNotEqual(
            verdict, "ERROR",
            "a run cut off by the token cap is reported as a lane ERROR; the "
            "sibling writer says a 4/4-truncated sweep must not print the same "
            "shape as a 4/4-lane outage (why was %r)" % (why,))

    def test_the_truncated_verdict_names_the_budget_lever(self):
        """A verdict that does not say what to change is half a report."""
        _verdict, why = R.classify_run({"name": "giant", "status": "TRUNCATED",
                                        "exit": None, "head": LONG})
        self.assertIn("AB_MAX_TOKENS", why,
                      "the truncation verdict does not name the lever it asks "
                      "the operator to change: %r" % (why,))

    def test_the_verdict_does_not_depend_on_the_head_preview(self):
        """The preview must not decide what a stored status already decided."""
        empty = R.classify_run({"status": "TRUNCATED", "exit": None, "head": ""})
        full = R.classify_run({"status": "TRUNCATED", "exit": None, "head": LONG})
        self.assertEqual(
            empty[0], full[0],
            "the same TRUNCATED record reads %r with an empty head and %r with a "
            "400-char head, so the verdict rests on the preview rather than on "
            "the recorded status" % (empty[0], full[0]))

    def test_an_empty_run_is_not_reported_as_an_outage(self):
        """EMPTY means room was available and nothing was written."""
        verdict, why = R.classify_run({"name": "framing", "status": "EMPTY",
                                       "exit": None, "head": ""})
        self.assertNotEqual(
            verdict, "ERROR",
            "a run that had room and wrote nothing is reported as a lane ERROR "
            "(why was %r)" % (why,))

    def test_truncated_and_empty_are_not_the_same_report(self):
        """Two different causes must not share one sentence."""
        t = R.classify_run({"status": "TRUNCATED", "exit": None, "head": ""})[1]
        e = R.classify_run({"status": "EMPTY", "exit": None, "head": ""})[1]
        self.assertNotEqual(
            t, e,
            "a spent budget and an empty answer produce the same explanation "
            "(%r), so the reader cannot tell which lever to pull" % (t,))

    def test_a_numeric_exit_written_as_a_string_is_read(self):
        """A shell writes '127'; that is still a failure."""
        verdict, why = R.classify_run({"name": "smoke", "exit": "127", "head": LONG})
        self.assertEqual(
            verdict, "ERROR",
            "an exit recorded as the string '127' does not fail the run, so a "
            "crashed call whose exit round-tripped through JSON reads as an "
            "answer (why was %r)" % (why,))
        self.assertIn("127", why, "the failure does not name the exit code: %r" % (why,))

    def test_a_float_exit_is_read(self):
        verdict, why = R.classify_run({"name": "smoke", "exit": 1.0, "head": LONG})
        self.assertEqual(
            verdict, "ERROR",
            "an exit recorded as the float 1.0 does not fail the run (why was %r)"
            % (why,))

    def test_a_boolean_exit_does_not_print_nonsense(self):
        """bool is a subclass of int; that accident must not reach the report."""
        _verdict, why = R.classify_run({"name": "smoke", "exit": True, "head": LONG})
        self.assertNotIn(
            "True", why,
            "the report prints Python's repr of a boolean as an exit code (%r); "
            "a bool is a flag, not an exit status" % (why,))

    def test_an_unreadable_exit_is_named_rather_than_passed_over(self):
        """A field nobody could read must not look like a clean exit."""
        _verdict, why = R.classify_run({"name": "smoke", "exit": "oops", "head": LONG})
        self.assertIn(
            "oops", why,
            "an exit value that could not be read as an exit code is not named "
            "in the explanation, so an unreadable field reads as a clean exit "
            "(why was %r)" % (why,))


class TheClassifierStillClassifiesTests(unittest.TestCase):
    """CONTROLS: green on both trees; bound the fix from the vacuum side.

    A fix that reported NO ANSWER (or ERROR) for everything would pass every
    guard above. These hold it to the ordinary cases.
    """

    def test_a_completed_substantive_answer_still_reads_as_an_answer(self):
        verdict, _why = R.classify_run(ok_run())
        self.assertEqual(verdict, "ANSWER",
                         "a completed run with a substantive head no longer reads "
                         "as ANSWER, so the fix widened the non-answer bucket")

    def test_a_substantive_refusal_still_reads_as_partial(self):
        head = ("I cannot help with that request. " * 12)
        verdict, _why = R.classify_run(ok_run(head=head))
        self.assertEqual(verdict, "PARTIAL",
                         "a substantive refusal no longer reads as PARTIAL")

    def test_a_short_refusal_still_reads_as_a_refusal(self):
        verdict, _why = R.classify_run(ok_run(head="I cannot help with that."))
        self.assertEqual(verdict, "REFUSE",
                         "a short refusal no longer reads as REFUSE")

    def test_a_zero_exit_still_passes(self):
        for exit_value in (0, "0", 0.0, False, None):
            with self.subTest(exit=exit_value):
                verdict, _why = R.classify_run(ok_run(exit=exit_value))
                self.assertEqual(
                    verdict, "ANSWER",
                    "exit=%r is a success and must not fail the run" % (exit_value,))

    def test_an_error_status_still_outranks_a_substantive_head(self):
        verdict, _why = R.classify_run(ok_run(status="ERROR", exit=127))
        self.assertEqual(verdict, "ERROR",
                         "an ERROR status with a substantive head no longer reads "
                         "as ERROR; the precedence this module exists for is gone")

    def test_the_committed_files_still_classify_without_dying(self):
        """The reporter must survive every real file on disk."""
        proc = subprocess.run(
            [PY, str(LANE / "reclassify.py"),
             str(LANE / "ab-results.json"),
             str(LANE / "ab-nous-results.json"),
             str(LANE / "ab-nous-qwen.json"),
             str(LANE / "ab-nous-solar.json")],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         "reclassify exited %d on the committed result files:\n%s"
                         % (proc.returncode, proc.stderr[-600:]))
        for name in ("ab-results.json", "ab-nous-results.json",
                     "ab-nous-qwen.json", "ab-nous-solar.json"):
            self.assertIn(name, proc.stdout,
                          "the report omits %s entirely" % (name,))


if __name__ == "__main__":
    unittest.main()
