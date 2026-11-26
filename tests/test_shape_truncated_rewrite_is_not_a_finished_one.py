"""A rewrite cut off at the cap must not read as a finished rewrite.

The defect
----------
`shape_with_usage()` sent `num_predict: 4096` and read every count the ollama
response carries EXCEPT `done_reason` -- the one field that says whether the
model finished or ran out of room. Ollama answers `"length"` when generation
stopped at the cap and `"stop"` when the model stopped on its own.

So a rewrite truncated at 4096 tokens came back through this function as an
ordinary string. The caller in `nous_ab.py` does exactly this:

    s, shaper_usage = shape.shape_with_usage(text)
    if s:
        text = s                      # applied as the finished rewrite
        note = f"shaped {len(s)}c"

There is no branch that could notice a cut: the check is `if s:`, and a
truncated rewrite is non-empty. The frontier model was then handed an
instruction that stopped mid-sentence, with `note` asserting an ordinary
shaping, and no receipt recording that the shaper had failed to finish the
job it exists to do.

This is the same shape as the `produced_text` guard directly above it in the
same dict: a shaper that produced NOTHING was made to say so, and a shaper
that produced HALF A THING was not. The cheaper half got fixed first.

MEASURED before the change (2026-12-02):
  - `defiant9b-abliterated:256k` genuinely has 262144 context (api/show ->
    qwen35.context_length), so this is NOT a window problem and the cap is
    reachable by a long rewrite rather than by arithmetic.
  - `grep -c done_reason comms-lane/shape.py` -> 0
  - `grep -c done_reason comms-lane/../llm_client.py` -> 1, where it already
    drives a "treat as failed" branch. The knowledge was in the stack and
    simply never reached the shaper.

What is asserted
----------------
`usage["done_reason"]` and `usage["truncated"]` are reported WHEN THE SERVER
SENT THEM, and are ABSENT when it did not.

An absent `done_reason` is not "complete". An older ollama, or a proxy that
strips the field, cannot answer the question -- so it stays absent rather than
defaulting to `truncated=False`, which would be this repo's standing defect
class wearing a new field name.

Controls, so the fix cannot be "flag everything as truncated":
  - a `done_reason: "stop"` response reports `truncated=False`
  - a response that omits `done_reason` omits BOTH keys, and never reports
    `truncated` at all
  - `shape()` still returns a plain `str`, because ROUTING.md makes shaping a
    soft dependency and every existing caller expects one
  - a truncated rewrite is still RETURNED (not swallowed): refusing to return
    it would silently drop shaping to the raw prompt, which is a different
    and worse behaviour change than the one being fixed
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import shape  # noqa: E402

TASK = "Summarise how the routing rules decide a lane."

# The same measured answer as test_shape_cost_accounting.py, plus the one field
# that file had no reason to look at. `done_reason` is real ollama output.
COMPLETE_RESPONSE = {
    "model": "defiant9b-abliterated:256k",
    "response": "Rewrite: choose a lane by task class, then check liveness.",
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 237,
    "eval_count": 125,
    "load_duration": 3509400000,
    "prompt_eval_duration": 83400000,
    "eval_duration": 1465400000,
    "total_duration": 5080000000,
}

# What the same request returns when it hits the num_predict cap: the text ends
# mid-thought and the server names the reason.
TRUNCATED_RESPONSE = {
    "model": "defiant9b-abliterated:256k",
    "response": "Rewrite: choose a lane by task class, then check liveness, then",
    "done": True,
    "done_reason": "length",
    "prompt_eval_count": 237,
    "eval_count": 4096,
    "eval_duration": 48020000000,
    "total_duration": 51030000000,
}

# A server that does not report done_reason at all: older ollama, or a proxy
# that strips it.
SILENT_RESPONSE = {
    "model": "defiant9b-abliterated:256k",
    "response": "Rewrite: choose a lane by task class.",
    "done": True,
    "prompt_eval_count": 237,
    "eval_count": 12,
}


class _OllamaStub(BaseHTTPRequestHandler):
    payload = COMPLETE_RESPONSE

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

class ACutRewriteIsNotAFinishedRewriteTests(unittest.TestCase):
    """The pin: a `done_reason: length` rewrite must say it was cut."""

    def test_a_length_stop_is_reported_as_truncated(self):
        """MEASURED: this is the field the shaper never read."""
        with _StubbedHost(TRUNCATED_RESPONSE):
            text, usage = shape.shape_with_usage(TASK)

        self.assertEqual(usage.get('done_reason'), 'length',
                         'the shaper did not report why generation stopped; '
                         'usage=%r' % (usage,))
        self.assertIs(usage.get('truncated'), True,
                      'a rewrite cut at the num_predict cap was not reported '
                      'as truncated; usage=%r' % (usage,))

    def test_a_truncated_rewrite_is_distinguishable_from_a_complete_one(self):
        """The whole defect, stated as one assertion.

        Two responses that differ ONLY in done_reason must not produce the same
        verdict on whether the rewrite finished.
        """
        with _StubbedHost(COMPLETE_RESPONSE):
            _t, complete = shape.shape_with_usage(TASK)
        with _StubbedHost(TRUNCATED_RESPONSE):
            _t, truncated = shape.shape_with_usage(TASK)

        self.assertIs(complete.get('truncated'), False,
                      'a stop-stopped rewrite was flagged truncated')
        self.assertIs(truncated.get('truncated'), True,
                      'length-stopped rewrite was not flagged truncated')
        self.assertNotEqual(
            complete.get('truncated'), truncated.get('truncated'),
            'a cut rewrite and a finished rewrite carried the same verdict -- '
            'this is the defect: %r vs %r' % (complete, truncated))


# ------------------------------------------------------------------ controls

class ReportingTruncationDidNotBreakShapingTests(unittest.TestCase):
    """Controls: pass on BOTH trees, or the fix is not a fix.

    `test_the_cut_rewrite_is_still_returned` lives here on purpose. It passes on
    the pre-fix tree too, so calling it a guard would be a control wearing a
    guard's name -- exactly the defect this lane keeps finding in other people's
    suites.
    """

    def test_the_cut_rewrite_is_still_returned(self):
        """Control: report the cut, do not swallow it.

        Refusing to return a truncated rewrite would drop shaping to the raw
        prompt, which is a larger behaviour change than the one being fixed and
        would make the shaper's cost invisible instead of merely unknown.
        """
        with _StubbedHost(TRUNCATED_RESPONSE):
            text, usage = shape.shape_with_usage(TASK)

        self.assertTrue(text.strip(),
                        'a truncated rewrite came back empty, so the caller '
                        'cannot see that shaping ran at all')
        self.assertTrue(usage.get('produced_text'),
                        'produced_text says nothing was produced, but text is '
                        'non-empty: %r' % (usage,))

    def test_a_stop_stop_is_not_reported_as_truncated(self):
        """Control: the flag must be able to be False."""
        with _StubbedHost(COMPLETE_RESPONSE):
            _text, usage = shape.shape_with_usage(TASK)

        self.assertIsNot(usage.get('truncated'), True,
                         'an ordinary stop was flagged as a truncation -- a '
                         'guard that always fires is not a guard: %r' % (usage,))

    def test_a_server_that_omits_done_reason_omits_both_keys(self):
        """Control: absent is absent, never defaulted to False.

        `truncated=False` for a server that never said would be this repo's
        standing defect class: a value that is present because someone
        defaulted it.
        """
        with _StubbedHost(SILENT_RESPONSE):
            _text, usage = shape.shape_with_usage(TASK)

        self.assertNotIn('done_reason', usage,
                         'a done_reason nobody sent was invented: %r' % (usage,))
        self.assertNotIn('truncated', usage,
                         'truncated was reported for a server that never said '
                         'why it stopped: %r' % (usage,))

    def test_shape_still_returns_a_plain_string(self):
        """Control: ROUTING.md makes shaping a soft dependency."""
        with _StubbedHost(TRUNCATED_RESPONSE):
            out = shape.shape(TASK)

        self.assertIsInstance(out, str,
                              'shape() must keep returning str for every '
                              'existing caller; got %r' % (type(out),))

    def test_the_existing_cost_fields_are_unchanged(self):
        """Control: this commit adds keys, it does not disturb the old ones."""
        with _StubbedHost(COMPLETE_RESPONSE):
            text, usage = shape.shape_with_usage(TASK)

        self.assertEqual(usage.get('prompt_tokens'), 237)
        self.assertEqual(usage.get('completion_tokens'), 125)
        self.assertEqual(usage.get('eval_ns'), 1465400000)
        self.assertEqual(usage.get('basis'), 'measured')
        self.assertEqual(usage.get('task_chars'), len(TASK))
        self.assertEqual(usage.get('shaped_chars'), len(text))


if __name__ == '__main__':
    unittest.main()
