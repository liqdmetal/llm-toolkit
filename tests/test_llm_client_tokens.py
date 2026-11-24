"""Pins for llm_client's token accounting, and the import crash beside it.

This stack's stated goal is to DECREASE token usage. That goal needs a counter
that can see the thing it counts, and llm_client.py could not:

DEFECT 1 -- the local lane reported a GUESS, not the measurement.
    Ollama /api/generate returns `prompt_eval_count` and `eval_count` in the
    very JSON body local_chat already parses. It read neither and reported
    `len(text) // 4` instead. MEASURED 2026-11-26 against a realistic body with
    prompt_eval_count=917, eval_count=43:

        reported prompt=0, completion=45
        measured prompt=917, completion=43

    Prompt cost is the number routing and shaping decisions are paid against,
    and it read as ZERO for every local call. Any "we shaped it and spent less"
    claim on this lane was unfalsifiable, because the counter was blind to the
    input entirely. `reference/gateway.py` already read the same two fields
    correctly, so this file held the second of TWO RULES FOR ONE VARIABLE.

DEFECT 2 -- an empty credential pool made the module unimportable.
    `_AUTH.get("credential_pool", {}).get("openrouter", [{}])[0]` indexed the
    default `[{}]`, but that default only applies when the KEY is absent. A key
    that is PRESENT and holds an empty list -- exactly what a host with no
    stored OpenRouter key has -- returned `[]`, and `[][0]` raised
    `IndexError: list index out of range` at IMPORT time. The whole client was
    unimportable on such a host, for a missing credential it was designed to
    tolerate.

WHY THE COUNTS HERE ARE ABSENT-VS-ZERO
    A counter that is missing means "not measured". A counter that is present
    and 0 is a MEASUREMENT of zero. Collapsing them is the same defect one
    layer down, so both directions are pinned, and `basis` states which one
    happened so a caller can refuse to bank an estimate.

CONTROLS
    A fix that reported hardware counters everywhere would be wrong too: a lane
    whose response omits the fields must keep the exact old arithmetic, or every
    historical receipt on this client changes meaning. Two controls prove that,
    plus that the model-still-loading failure path is untouched.

    python -m pytest tests/test_llm_client_tokens.py -q
"""
import importlib.util
import io
import itertools
import json
import os
import sys
import tempfile

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "llm_client.py")
_counter = itertools.count()

# A body shaped like a real Ollama /api/generate response.
BODY_MEASURED = {
    "model": "qwen3.8-uncensored:27b",
    "response": "The quick brown fox jumps over the lazy dog. " * 4,
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 917,
    "eval_count": 43,
}


class _FakeResp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _load(auth_doc, api_key=None):
    """Import llm_client against a THROWAWAY HERMES_HOME.

    The real ~/.hermes/auth.json is never read: this module is loaded with
    HERMES_HOME pointed at a temp dir. That also makes the empty-pool crash
    reproducible instead of depending on what the operator happens to have
    stored.
    """
    home = tempfile.mkdtemp(prefix="llmtk-auth-")
    with open(os.path.join(home, "auth.json"), "w", encoding="utf-8") as fh:
        json.dump(auth_doc, fh)

    saved = {k: os.environ.get(k) for k in ("HERMES_HOME", "OPENROUTER_API_KEY")}
    os.environ["HERMES_HOME"] = home
    if api_key is None:
        os.environ.pop("OPENROUTER_API_KEY", None)
    else:
        os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        name = "llm_client_under_test_%d" % next(_counter)
        spec = importlib.util.spec_from_file_location(name, SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _chat(mod, body):
    """Drive local_chat over a canned response body. No network, no model."""
    original = mod.urllib.request.urlopen
    mod.urllib.request.urlopen = lambda req, timeout=None: _FakeResp(
        json.dumps(body).encode())
    try:
        return mod.local_chat([{"role": "user", "content": "hi"}])
    finally:
        mod.urllib.request.urlopen = original


# ---------------------------------------------------------------- the defects


def test_guard_prompt_tokens_are_the_measured_count():
    """GUARD. Pre-fix this is 0: a byte heuristic over an EMPTY prompt string.

    The prompt local_chat builds here is `hi`, four bytes, so len//4 is 0 while
    Ollama measured 917 tokens of context. That is the whole defect in one
    number: the counter reported no input at all.
    """
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="probe")
    r = _chat(mod, BODY_MEASURED)
    assert r["ok"] is True
    assert r["tokens"]["prompt"] == BODY_MEASURED["prompt_eval_count"]


def test_guard_completion_tokens_are_the_measured_count():
    """GUARD. Pre-fix this is 45 (len//4 of a 176-char answer), here 43."""
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="probe")
    r = _chat(mod, BODY_MEASURED)
    assert r["tokens"]["completion"] == BODY_MEASURED["eval_count"]


def test_guard_a_measured_zero_is_not_an_absent_counter():
    """GUARD. ABSENT and ZERO must not collapse.

    A response that positively reports prompt_eval_count=0 measured zero tokens
    of prompt. Pre-fix the same body reported len(prompt)//4, which for the
    non-empty prompt built below is greater than zero -- the two states were
    indistinguishable and both were a guess.
    """
    body = dict(BODY_MEASURED, prompt_eval_count=0, eval_count=0)
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="probe")
    r = _chat(mod, body)
    assert r["tokens"]["prompt"] == 0
    assert r["tokens"]["completion"] == 0


def test_guard_the_report_states_whether_it_measured():
    """GUARD. A counter nobody can trust the provenance of is the original bug.

    `basis` is read through .get() so a pre-fix tree reports "<absent>" rather
    than raising KeyError: a missing key would bound no behaviour, and this pin
    is about what the client CLAIMS, not whether a field exists.
    """
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="probe")
    r = _chat(mod, BODY_MEASURED)
    assert r["tokens"].get("basis", "<absent>") == "measured"


def test_guard_an_empty_openrouter_pool_does_not_break_import():
    """GUARD. Pre-fix: IndexError: list index out of range, at import.

    A credential pool present-with-zero-entries is what a host with no stored
    OpenRouter key has. It made the entire module unimportable, so every lane in
    the client was gone because of one missing optional credential.
    """
    mod = _load({"credential_pool": {"openrouter": []}}, api_key=None)
    assert mod._OPENROUTER_KEY == ""


# ------------------------------------------------- the arithmetic that must not


def test_control_absent_counters_keep_the_old_arithmetic():
    """CONTROL -- passes on BOTH trees. A lane that omits the fields is not
    silently upgraded to a hardware reading, and every historical receipt on
    this client keeps its meaning: len//4, byte for byte."""
    body = {k: v for k, v in BODY_MEASURED.items()
            if k not in ("prompt_eval_count", "eval_count")}
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="probe")
    r = _chat(mod, body)
    assert r["tokens"]["prompt"] == 0
    assert r["tokens"]["completion"] == len(BODY_MEASURED["response"]) // 4


def test_control_a_model_still_loading_still_fails_through():
    """CONTROL -- passes on BOTH trees. The load-in-progress path returns
    ok=False with no tokens at all, so the router still falls through to aux
    instead of banking a hallucinated zero against a model that never ran."""
    body = {"response": "", "done": True, "done_reason": "load",
            "prompt_eval_count": 917, "eval_count": 0}
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="probe")
    r = _chat(mod, body)
    assert r["ok"] is False
    assert "tokens" not in r


def test_control_the_env_key_still_wins_over_the_pool():
    """CONTROL -- passes on BOTH trees. The env override is intact; only the
    index into an empty pool changed."""
    mod = _load({"credential_pool": {"openrouter": []}}, api_key="env-wins")
    assert mod._OPENROUTER_KEY == "env-wins"


def test_control_a_stored_pool_entry_is_still_read():
    """CONTROL -- passes on BOTH trees. Non-empty pools keep working, so the fix
    is a guard on the empty case and not a bypass of the credential store."""
    mod = _load({"credential_pool": {"openrouter": [{"access_token": "pool-key"}]}},
                api_key=None)
    assert mod._OPENROUTER_KEY == "pool-key"
