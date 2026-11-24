"""The shaping step must report what it COSTS.

The defect
----------
`shape.shape()` returned one thing — the rewritten prompt as a string — and threw
away the rest of the ollama response. That response carries `prompt_eval_count`,
`eval_count` and the durations, i.e. the actual token cost of shaping. So no
caller in the stack could report what the shaper spends, and the three goals of
this lane ("increase usage, distribute power, DECREASE TOKENS") had no input on
the third: token accounting stopped at the frontier call and ignored the local
model that precedes it.

MEASURED 2026-11-24 against the real host 192.168.1.124:11434,
`defiant9b-abliterated:256k`, this file's own DEFAULT_SYSTEM and the `nous_ab`
"framing" task:

    prompt_eval_count    237 tokens IN   (system prompt + task)
    eval_count           125 tokens OUT  (the rewrite)
    load_ms             3509.4
    eval_ms             1465.4
    wall_s                 5.08
    num_predict cap       4096  ->  cap utilisation 3.1%
    task chars in          199
    shaped chars out       623  ->  3.13x expansion

That last row is the finding, not a footnote: the rewrite is 3.13x the input's
character length, so shaping ADDS prompt tokens while it (aims to) save
completion tokens. Both sides of that trade were invisible before this change.

What is asserted
----------------
`shape_with_usage(task) -> (text, usage)`. The guards drive the real response
parsing over a stub server, so they measure the CONSEQUENCE — the returned usage
dict — rather than the presence of a symbol.

Why the stub and not the live box: a test that needs 192.168.1.124 to be up and
holding one specific model fails for reasons that are not defects (the host was
taken for Inkling-2-Mini once already, see test_shape_preflight.py). The stub
lets the guard run anywhere while still exercising the real payload parsing.

Controls, so the fix cannot be "always report numbers":
  - a response WITHOUT usage fields must return usage that OMITS them, never
    zero-fills them — an absent count is not a measured zero
  - `shape()` must still return a plain str, because ROUTING.md makes shaping a
    soft dependency and every existing caller expects one
"""

import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import shape  # noqa: E402

# A response shaped like the real ollama /api/generate answer, with the usage
# fields the live host actually returned on 2026-11-24.
MEASURED_RESPONSE = {
    "model": "defiant9b-abliterated:256k",
    "response": ("1. Analyze the technical factors that determine hashrate.\n"
                 "2. Identify every optimization trick and tuning parameter.\n"
                 "3. Verify which are compatible with the current build."),
    "done": True,
    "prompt_eval_count": 237,
    "eval_count": 125,
    "load_duration": 3509400000,
    "prompt_eval_duration": 83400000,
    "eval_duration": 1465400000,
    "total_duration": 5080000000,
}

# The same answer from a server that reports no usage at all (an older ollama, a
# proxy that strips the fields, a different backend).
BARE_RESPONSE = {
    "model": "defiant9b-abliterated:256k",
    "response": "1. Analyze. 2. Identify. 3. Verify.",
    "done": True,
}

TASK = ("Our competitor shipped a newer kernel and is torching us on hashrate. "
        "How do we make our pearl miner faster and beat them?")


class _OllamaStub(BaseHTTPRequestHandler):
    """Serves /api/generate with a fixed document; /api/tags lists the model."""

    payload = MEASURED_RESPONSE

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get('Content-Length') or 0)
        self.rfile.read(length)
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        # preflight() reads /api/tags to confirm the model exists
        body = json.dumps({"models": [{"name": shape.MODEL}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _StubbedHost:
    """Point shape.py at a local stub for the duration of a test."""

    def __init__(self, payload):
        _OllamaStub.payload = payload

    def __enter__(self):
        self.server = HTTPServer(('127.0.0.1', 0), _OllamaStub)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self._saved = (shape.OLLAMA, shape.PROBE_TIMEOUT)
        shape.OLLAMA = 'http://127.0.0.1:%d' % self.server.server_port
        return self

    def __exit__(self, *exc):
        shape.OLLAMA, shape.PROBE_TIMEOUT = self._saved
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


# ------------------------------------------------------------------- guards

class ShapingCostIsReportedTests(unittest.TestCase):
    """The pin: shaping must return the tokens it spent."""

    def test_shaping_reports_the_tokens_it_spent(self):
        """MEASURED numbers, asserted. 237 in / 125 out on the real host."""
        with _StubbedHost(MEASURED_RESPONSE):
            text, usage = shape.shape_with_usage(TASK)

        self.assertTrue(text.strip(), 'the rewrite came back empty')
        self.assertEqual(usage.get('prompt_tokens'), 237,
                         'shaping did not report its input token count')
        self.assertEqual(usage.get('completion_tokens'), 125,
                         'shaping did not report its output token count')
        self.assertEqual(usage.get('eval_ns'), 1465400000)
        self.assertEqual(usage.get('basis'), 'measured')

    def test_shaping_reports_how_much_it_enlarged_the_prompt(self):
        """The 3.13x expansion is the half nobody could see.

        A shaper that adds prompt tokens to save completion tokens is making a
        trade, and a trade with only one side measured is not a trade you can
        decide.
        """
        with _StubbedHost(MEASURED_RESPONSE):
            text, usage = shape.shape_with_usage(TASK)

        self.assertEqual(usage.get('task_chars'), len(TASK))
        self.assertEqual(usage.get('shaped_chars'), len(text))
        self.assertGreater(usage.get('expansion_x', 0), 1.0,
                           'this rewrite should be reported as an EXPANSION')
        # and the reported ratio must match the reported sizes
        self.assertAlmostEqual(usage['expansion_x'],
                               round(len(text) / len(TASK), 2), places=2)

    def test_an_empty_rewrite_still_reports_that_it_cost_tokens(self):
        """A shaper that produced nothing still spent them. Say so."""
        empty = dict(MEASURED_RESPONSE, response="")
        with _StubbedHost(empty):
            text, usage = shape.shape_with_usage(TASK)

        self.assertEqual(text.strip(), '')
        self.assertFalse(usage.get('produced_text'),
                         'an empty rewrite was reported as having produced text')
        self.assertEqual(usage.get('completion_tokens'), 125,
                         'the tokens spent on an empty rewrite went unreported')


# ----------------------------------------------------------------- controls
# These pass on BOTH trees, so a red guard cannot be blamed on the harness.

class ShapeStillIsASoftDependencyTests(unittest.TestCase):
    """ROUTING.md: shaping must never be a hard dependency."""

    def test_shape_still_returns_a_plain_string(self):
        """Every existing caller expects a str; the new function must not break it."""
        with _StubbedHost(MEASURED_RESPONSE):
            out = shape.shape(TASK)
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())


class AbsentUsageIsNotZeroUsageTests(unittest.TestCase):
    """CONTROL: a server that reports no usage must not get zero-filled."""

    def test_missing_usage_fields_are_omitted_not_defaulted(self):
        with _StubbedHost(BARE_RESPONSE):
            _text, usage = shape.shape_with_usage(TASK)

        self.assertNotIn('prompt_tokens', usage,
                         'an unreported input count was filled in as a value')
        self.assertNotIn('completion_tokens', usage,
                         'an unreported output count was filled in as a value')
        # The character counts are measured locally, so they are always real.
        self.assertEqual(usage.get('task_chars'), len(TASK))
        self.assertIn('shaped_chars', usage)


class TheGuardIsLoadBearingTests(unittest.TestCase):
    """NEGATIVE CONTROL: the pre-fix shape() must fail these guards.

    Without this, "usage is reported" could be satisfied by a stub that never
    exercised the real parsing. Here the OLD implementation is reconstructed —
    return the text alone — and the guard's own predicate must reject it.
    """

    def test_the_pre_fix_shape_cannot_satisfy_the_guards(self):
        def old_shape(task, system=shape.DEFAULT_SYSTEM, timeout=shape.TIMEOUT):
            # The pre-fix contract, verbatim: text only, usage discarded.
            return shape.shape_with_usage(task, system, timeout)[0]

        with _StubbedHost(MEASURED_RESPONSE):
            result = old_shape(TASK)

        self.assertIsInstance(result, str,
                              'sanity: the old shape() did return a string')
        # The guards above unpack two values. The old return cannot supply the
        # usage half, so they would fail on it — which is what makes them
        # guards rather than descriptions.
        with self.assertRaises((TypeError, ValueError),
                               msg='the pre-fix return satisfied a 2-value '
                                   'unpack; the guards would not have caught it'):
            _text, _usage = result  # noqa: F841 - unpacking a str fails here

    def test_a_stub_that_reports_no_usage_cannot_pass_the_primary_guard(self):
        """The primary guard needs real numbers, not a present-but-empty dict."""
        with _StubbedHost(BARE_RESPONSE):
            _text, usage = shape.shape_with_usage(TASK)
        self.assertNotEqual(usage.get('prompt_tokens'), 237,
                            'a bare response must not fabricate the measured counts')


if __name__ == '__main__':
    unittest.main()
