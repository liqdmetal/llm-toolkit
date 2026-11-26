"""A server that did not report a timing is not a lane measured at zero tok/s.

Standing defect class in this stack: **an absent count is not a measured zero.**
``shape.py`` states the rule for its usage dict ("keys are absent when the server
did not report them, never zero-filled"), ``fleet_probe`` states it for residency
("a name marked unknown rather than inventing one"), and ``bench_local.py`` --
the tool whose receipt is what a routing decision is actually made on -- breaks
it in the one place a reader is least able to notice, because the number it
invents is a SPEED.

MEASURED at HEAD 7d97af1
------------------------
Through the real ``ollama_once`` / ``openai_once`` with ``_post`` replaced (no
network, no model, nothing generated):

    ollama payload                          decode_tok_s   decode_tokens  prefill_tokens
    {"eval_count":100,"eval_duration":2s,
     "prompt_eval_count":20,...}                 50.0          100             20
    {"eval_count":100,"eval_duration":2s}        50.0          100      ->    0
    {"eval_duration":2s}        (count absent) -> 0.0       ->  0            0
    {}                                            None      ->  0            0

    openai payload                          decode_tokens
    {"usage":{"completion_tokens":200,...}}      200
    {}                                       ->    0

Four fabricated zeros from one ``or 0`` each:

1. ``dec_n = d.get("eval_count") or 0`` -- an absent count becomes 0 tokens, so
   ``0 / 2.0s`` is a **0.0 tok/s** row. This is the sharp one, because the
   renderer deliberately prints a measured zero (``is not None``, with the
   comment "a measured 0.0 tok/s is a measurement and must not render as the
   same dash a missing sample renders as"). That decision is right and stays;
   what was wrong is that an ABSENT count was being fed into it as a
   measurement. So the receipt says a lane decodes at zero tokens per second
   when the lane answered and simply did not report its counters -- a router
   reading that receipt skips a perfectly good lane, which is the expensive
   fall-through to a billed lane this whole stack exists to prevent, arrived at
   THROUGH the receipt.
2. ``pre_n = d.get("prompt_eval_count") or 0`` -- ``prefill_tokens: 0``.
3. ``"decode_tokens": dec_n`` / ``"decode_tokens": dec_n`` -- the fabricated 0
   is also copied into the raw count column, so the receipt's token totals are
   wrong and not merely its rate.
4. ``(d.get("load_duration") or 0) / 1e9`` -- a model that was never reported as
   loaded becomes ``load_s: 0.0``, which reads as an instantaneous load. A real
   load is 3.5 s on this fleet.

``openai_once`` has the same shape on ``usage.get("completion_tokens") or 0``:
a response with no ``usage`` block reports ``decode_tokens: 0``.

What is asserted
----------------
An omitted field stays absent (``None``) and does not become 0, in the rate, in
the raw counts, and in the load time. The control half is what stops the fix
going too far: a server that reports ``eval_count: 0`` is a real zero and must
still render ``0.0`` -- otherwise the repair would trade one lie (invented zero)
for another (an honest zero hidden as missing), and the renderer's deliberate
measured-zero branch would become dead code.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'comms-lane'))

import bench_local  # noqa: E402

assert Path(bench_local.__file__).resolve() == (ROOT / 'comms-lane' / 'bench_local.py').resolve(), \
    'imported the wrong bench_local: %s' % bench_local.__file__

TWO_S = 2_000_000_000
TENTH_S = 100_000_000


class _PostFake:
    """Replace ``bench_local._post`` for one call and restore it."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        self._real = bench_local._post
        bench_local._post = lambda *a, **kw: self.payload
        return self

    def __exit__(self, *a):
        bench_local._post = self._real
        return False


def _ollama(payload):
    with _PostFake(payload):
        return bench_local.ollama_once('http://stub', 'stub-model')


def _openai(payload):
    with _PostFake(payload):
        return bench_local.openai_once('http://stub', 'stub-model')


FULL_OLLAMA = {
    'eval_count': 100, 'eval_duration': TWO_S,
    'prompt_eval_count': 20, 'prompt_eval_duration': TENTH_S,
    'load_duration': int(3.5e9),
}


class GuardsAbsentIsNotZero(unittest.TestCase):
    def test_a_missing_token_count_is_not_zero_tokens_per_second(self):
        """eval_count absent, duration present: the rate is unknown, not 0.0."""
        row = _ollama({'eval_duration': TWO_S})
        self.assertNotEqual(
            row['decode_tok_s'], 0.0,
            'a server that did not report eval_count was recorded as decoding '
            'at 0.0 tok/s, and the table prints a measured 0.0 rather than a '
            'dash -- so the receipt states a speed the server never gave')

    def test_a_missing_token_count_is_not_zero_tokens(self):
        """The same absent count must not become 0 in the raw token column."""
        row = _ollama({'eval_duration': TWO_S})
        self.assertNotEqual(
            row['decode_tokens'], 0,
            'an absent eval_count was copied into decode_tokens as 0, so the '
            "receipt's token totals count a figure the server never sent")

    def test_a_missing_prefill_count_is_not_zero_prefill_tokens(self):
        row = _ollama({'eval_count': 100, 'eval_duration': TWO_S})
        self.assertNotEqual(
            row['prefill_tokens'], 0,
            'a server that did not report prompt_eval_count was recorded as '
            'prefilling 0 tokens, which asserts the prompt cost nothing')

    def test_a_missing_load_duration_is_not_an_instant_load(self):
        row = _ollama({'eval_count': 100, 'eval_duration': TWO_S})
        self.assertNotEqual(
            row['load_s'], 0.0,
            'a model whose load_duration was never reported was recorded as '
            'loading in 0.0 s -- a real load is seconds on this fleet')

    def test_an_openai_response_with_no_usage_is_not_zero_tokens(self):
        row = _openai({})
        self.assertNotEqual(
            row['decode_tokens'], 0,
            'a hosted reply carrying no usage block was recorded as 0 '
            'completion tokens instead of an unreported count')

    def test_a_real_measured_zero_reaches_the_receipt(self):
        """The inverse defect: `measure()` discards an honest 0.0 tok/s.

        MEASURED pre-fix: a server reporting ``eval_count: 0`` with a real
        duration produces a sample of ``0.0``, and ``measure()`` then filters it
        out with ``if s.get("decode_tok_s")`` -- 0.0 is falsy -- so the row
        carries NO ``decode_tok_s`` at all and renders as the ``—`` dash.
        The renderer's own comment says a measured 0.0 must not render as the
        dash a missing sample renders as; ``measure()`` made that unreachable.

        So the two defects are the same thumb on two sides of one scale: the
        absent count was promoted to a measured 0.0, and the real measured 0.0
        was demoted to absent. Fixing only the first would leave the second.
        """
        with _PostFake({'eval_count': 0, 'eval_duration': TWO_S,
                        'prompt_eval_count': 0,
                        'prompt_eval_duration': TENTH_S}):
            row = bench_local.measure(('slow', 'http://stub', 'm', 'ollama'), 2)
        self.assertIn(
            'decode_tok_s', row,
            'a lane that really decoded 0 tokens in 2.0 s has no decode_tok_s '
            'in its receipt: measure() dropped the honest 0.0 as falsy, so an '
            'answered-but-slow lane is indistinguishable from one that was '
            'never measured')
        self.assertEqual(row['decode_tok_s'], 0.0)


class ControlsBothTrees(unittest.TestCase):
    """These must pass on the pre-fix tree AND on the fixed one."""

    def test_the_table_shows_a_dash_for_a_lane_that_reported_no_count(self):
        """End of the wire: an unreported speed is a dash, never a number."""
        with _PostFake({'eval_duration': TWO_S}):
            row = bench_local.measure(('silent', 'http://stub', 'm', 'ollama'), 1)
        table = bench_local.render([row])
        self.assertNotIn(
            '| silent | m | 0.0 |', table,
            'the receipt rendered a lane as 0.0 tok/s when the server omitted '
            'its token count -- the table is what a routing decision reads')
        self.assertIn('| silent | m | — |', table,
                      'a lane that reported no timing must render as a dash')

    def test_full_timings_are_reported_exactly(self):
        row = _ollama(FULL_OLLAMA)
        self.assertEqual(row['decode_tok_s'], 50.0)
        self.assertEqual(row['prefill_tok_s'], 200.0)
        self.assertEqual(row['decode_tokens'], 100)
        self.assertEqual(row['prefill_tokens'], 20)
        self.assertEqual(row['load_s'], 3.5)

    def test_a_real_zero_count_stays_a_measured_zero(self):
        """eval_count reported as 0 is a measurement and must remain 0.0."""
        row = _ollama({'eval_count': 0, 'eval_duration': TWO_S,
                       'prompt_eval_count': 0, 'prompt_eval_duration': TENTH_S})
        self.assertEqual(
            row['decode_tok_s'], 0.0,
            'an honest 0 tok/s was hidden as missing -- the repair must not '
            'trade an invented zero for a concealed real one')
        self.assertEqual(row['decode_tokens'], 0)
        self.assertEqual(row['prefill_tokens'], 0)

    def test_a_measured_zero_still_renders_as_a_number(self):
        """The renderer's measured-zero branch must stay reachable."""
        table = bench_local.render([{
            'lane': 'slow', 'model': 'm', 'kind': 'ollama',
            'decode_tok_s': 0.0, 'decode_tokens': 0, 'reps': 1,
            'residency_state': 'unprobed',
        }])
        self.assertIn('| slow | m | 0.0 |', table,
                      'a measured 0.0 tok/s must still print, not a dash')

    def test_a_missing_duration_leaves_the_rate_absent(self):
        """eval_count present, duration absent: no rate can be computed."""
        row = _ollama({'eval_count': 100, 'prompt_eval_count': 20})
        self.assertIsNone(row['decode_tok_s'])

    def test_openai_full_usage_is_reported_exactly(self):
        row = _openai({'usage': {'completion_tokens': 200,
                                 'prompt_tokens': 30, 'cost': 0.1}})
        self.assertEqual(row['decode_tokens'], 200)
        self.assertEqual(row['prefill_tokens'], 30)
        self.assertEqual(row['cost'], 0.1)

    def test_an_absent_cost_is_still_absent(self):
        row = _openai({'usage': {'completion_tokens': 200, 'prompt_tokens': 30}})
        self.assertIsNone(row['cost'])


if __name__ == '__main__':
    unittest.main()
