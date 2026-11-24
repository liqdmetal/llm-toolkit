"""Pins for shape.py's reachability preflight.

Why this exists: on 2026-11-24 Hermes took the 3090 (192.168.1.124) to serve
Inkling-2-Mini, evicting defiant9b from VRAM for the duration. shape.py had no
reachability check, so every shaped prompt would open a generation request with
a 180 s timeout and block until that expired — to learn something a TCP connect
answers in milliseconds.

The guard is only worth having if it is cheap AND if its absence is expensive.
test_preflight_runs_fast is therefore paired with a control that measures what
the unguarded path costs, so a preflight that silently stopped running would
show up as a slow test rather than passing quietly.

    python -m pytest tests/test_shape_preflight.py -q
"""
import os
import socket
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comms-lane"))

import shape  # noqa: E402


def _dead_port() -> int:
    """A port nothing is listening on, obtained by binding then releasing.

    Not a hardcoded number: a stray service on a fixed port would turn every
    assertion here into a false pass. Binding an ephemeral port and closing it
    means the OS just handed it out as free.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def test_host_port_parses_default_and_override():
    assert shape._host_port("http://192.168.1.124:11434") == ("192.168.1.124", 11434)
    assert shape._host_port("http://example.invalid") == ("example.invalid", 80)
    assert shape._host_port("https://example.invalid") == ("example.invalid", 443)


def test_preflight_passes_when_something_listens():
    """Positive control: a live listener must NOT raise.

    Without this, a preflight that always raised would satisfy every
    unreachable-host assertion in this file and silently disable shaping
    fleet-wide.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        shape.preflight(f"http://127.0.0.1:{port}", timeout=2.0)
    finally:
        srv.close()


def test_preflight_raises_on_closed_port():
    port = _dead_port()
    with pytest.raises(shape.ShaperUnavailable) as ei:
        shape.preflight(f"http://127.0.0.1:{port}", timeout=1.0)
    assert "unreachable" in str(ei.value)


def test_shape_skips_before_generating_when_host_is_down(monkeypatch):
    """The real requirement: shape() must not attempt a generation.

    A generation attempt against a dead box is what burned the 180 s timeout.
    Asserting the exception type proves the refusal happens BEFORE the HTTP
    layer, not after it.
    """
    monkeypatch.setattr(shape, "OLLAMA", f"http://127.0.0.1:{_dead_port()}")
    monkeypatch.setattr(shape, "PROBE_TIMEOUT", 1.0)
    monkeypatch.delenv("SHAPE_DISABLE", raising=False)
    with pytest.raises(shape.ShaperUnavailable):
        shape.shape("rewrite this task")


def test_shape_disable_env_short_circuits(monkeypatch):
    """A reserved box must be declareable off without touching the network.

    Pointed at a LIVE port here, so the only reason to refuse is the env var.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        monkeypatch.setattr(shape, "OLLAMA", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("SHAPE_DISABLE", "1")
        assert shape.enabled() is False
        with pytest.raises(shape.ShaperUnavailable) as ei:
            shape.shape("rewrite this task")
        assert "disabled" in str(ei.value)
    finally:
        srv.close()


def test_preflight_is_fast_against_a_blackholed_host():
    """THE CONTROL: proves the guard is load-bearing, not decorative.

    A closed port on 127.0.0.1 is refused by the OS instantly, so it cannot
    distinguish a guarded path from an unguarded one — both look fast and the
    test would pass with the guard deleted. TEST-NET-1 (192.0.2.0/24, RFC 5737)
    is guaranteed unroutable, so a connect there does not refuse: it sits until
    the timeout expires. That is the only shape in which the probe budget is
    actually binding, i.e. the shape a reserved or firewalled 3090 presents.

    The unguarded path would spend TIMEOUT (180 s) here. The assertion is that
    the guarded path returns within a small multiple of the probe budget, so a
    preflight that silently stopped running fails this test instead of passing
    quietly.
    """
    probe = 0.5
    t0 = time.monotonic()
    with pytest.raises(shape.ShaperUnavailable):
        shape.preflight("http://192.0.2.1:11434", timeout=probe)
    guarded = time.monotonic() - t0

    assert guarded < probe * 6, f"preflight took {guarded:.2f}s against a {probe}s budget"
    assert guarded < shape.TIMEOUT / 10, "preflight must cost far less than the generation timeout"


def test_probe_budget_is_much_smaller_than_generation_budget():
    """Pins the two budgets apart so the escape hatch cannot be quietly widened.

    This is the assertion that would have caught the original defect: with no
    probe budget at all, every shape() call inherits the generation timeout.
    """
    assert shape.PROBE_TIMEOUT <= 5.0
    assert shape.TIMEOUT >= 60
    assert shape.PROBE_TIMEOUT * 10 < shape.TIMEOUT
