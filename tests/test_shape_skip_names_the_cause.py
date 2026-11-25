"""A shaping skip must not name a cause the server never reported.

Why this exists
---------------
The shaping lane is a soft dependency: when it cannot run, the caller uses its
own prompt, so the skip message is the only evidence an operator ever sees. The
load-failure skip named ONE cause unconditionally:

    SHAPE_SKIP: model 'defiant9b-abliterated:256k' is listed on
    192.168.1.124:11434 but could not be loaded (HTTP 500); that box has no room
    for it right now

MEASURED 2026-11-28: three genuinely different 500 bodies -- CUDA out of
memory, a corrupt GGUF tensor, an unknown model architecture -- all classified
as load failures through ``_is_model_load_failure`` and all produced that
identical sentence. An operator holding a corrupt file was told the card was
full, and would go hunting for VRAM that was never the problem.

What makes this more than pedantry: the 3090's ``/api/ps`` was EMPTY for the
whole measurement -- no model loaded, nothing holding the card -- while the same
message asserted the box had no room. The cause is now derived from the body the
server actually returned, or the message says the server stated none.

The two constraints that fight here
-----------------------------------
ROUTING.md rule 1 needs a listed-but-unloadable model to be a SKIP, not a hard
error, or every full card becomes an outage. That is ``_is_model_load_failure``
and it must keep its wide net. Separately, a cause must not be invented. So the
classification stays broad and only the MESSAGE stops asserting a cause: these
pins hold both, which is why the controls assert the skip still happens for each
distinct cause.

Order matters in the derivation: the real CUDA OOM body also carries
``llama-server process has terminated``, which is the symptom rather than the
cause, so the memory test has to win.
"""
import sys
from pathlib import Path

import pytest

LANE = Path(__file__).resolve().parents[1] / "comms-lane"
sys.path.insert(0, str(LANE))

import shape  # noqa: E402

OOM_BODY = (
    '{"error":"llama-server process has terminated: exit status 1: '
    "ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed (out of memory), "
    'returning 0/0 cudaMalloc failed: out of memory"}'
)
CORRUPT_BODY = (
    '{"error":"unable to load model: tensor blk.0.attn_q.weight data is not '
    'within the file bounds, model is corrupted"}'
)
ARCH_BODY = (
    '{"error":"failed to load model: unknown model architecture: \'llama4\'"}'
)
BARE_TERMINATED = '{"error":"llama-server process has terminated: exit status 1"}'

DISTINCT_BODIES = {"oom": OOM_BODY,
                   "corrupt": CORRUPT_BODY,
                   "architecture": ARCH_BODY}


# --- guards: each fails on the pre-repair tree -----------------------------

def test_each_load_failure_states_its_own_cause():
    """Three different failures must not share one sentence."""
    causes = {name: shape._load_failure_cause(body)
              for name, body in DISTINCT_BODIES.items()}
    assert len(set(causes.values())) == 3, causes


def test_a_corrupt_model_is_not_reported_as_a_full_card():
    """The concrete operator misdirection this repair removes."""
    cause = shape._load_failure_cause(CORRUPT_BODY)
    assert "memory" not in cause.lower(), cause
    assert "corrupt" in cause.lower(), cause


def test_an_unsupported_architecture_is_not_reported_as_a_full_card():
    cause = shape._load_failure_cause(ARCH_BODY)
    assert "memory" not in cause.lower(), cause
    assert "architecture" in cause.lower(), cause


def test_cuda_oom_names_memory_and_not_the_symptom():
    """The OOM body carries 'process has terminated' too; cause beats symptom."""
    cause = shape._load_failure_cause(OOM_BODY)
    assert "memory" in cause.lower(), cause
    assert "terminated" not in cause.lower(), cause


def test_a_causeless_failure_says_the_server_stated_no_cause():
    """Not knowing is a reportable state; guessing is not."""
    cause = shape._load_failure_cause(BARE_TERMINATED)
    assert "no cause" in cause.lower(), cause
    assert "memory" not in cause.lower(), cause


def test_the_skip_message_carries_the_derived_cause_not_a_fixed_one(monkeypatch):
    """End to end through the real raise site, not just the helper.

    The stub server used by test_shape_load_failure is the better harness, but
    this test needs the 500 body to reach the raise site verbatim, so urlopen is
    replaced with a real HTTPError carrying a readable body. A subclass with
    wrong constructor args raises TypeError inside the code under test and would
    "pass" a is-it-raised check for the wrong reason.
    """
    import io
    import urllib.error

    monkeypatch.setattr(shape, "MODEL", "defiant9b-abliterated:256k")
    monkeypatch.setattr(shape, "_installed_models",
                        lambda *_a, **_k: {"defiant9b-abliterated:256k"})

    def fake_urlopen(req, *_a, **_k):
        raise urllib.error.HTTPError(
            url=getattr(req, "full_url", "http://stub/api/generate"),
            code=500, msg="Internal Server Error", hdrs=None,
            fp=io.BytesIO(CORRUPT_BODY.encode()))

    monkeypatch.setattr(shape.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(shape.ShaperUnavailable) as ei:
        shape.shape("summarise this log")
    msg = str(ei.value)
    assert "corrupt" in msg.lower(), msg
    assert "no room" not in msg.lower(), msg


# --- controls: each passes on BOTH trees -----------------------------------

def test_control_a_load_failure_is_still_a_skip_not_an_error():
    """ROUTING.md rule 1 unchanged: the wide net must keep every cause."""
    for name, body in DISTINCT_BODIES.items():
        assert shape._is_model_load_failure(body) is True, name


def test_control_an_unrelated_500_is_still_raised():
    """A server bug must not become silent capacity loss."""
    assert shape._is_model_load_failure("rate limit exceeded") is False
    assert shape._is_model_load_failure("") is False
