"""A sweep to a card printed no lane count at all.

`nous_ab.py`'s summary gained a lane sentence on 2026-11-28 (`aa8939e`). It asked
a two-valued question -- hosted or not -- and both its branches required the
hosted count to be non-zero:

    hosted = sum(1 for _ in results["runs"] if _is_hosted(ENDPOINT))
    if hosted == len(results["runs"]):   print("all N ... HOSTED; 0 local GPU(s)")
    elif hosted:                          print("H/N hosted, N-H on a local lane")

MEASURED at aa8939e, driving the real `main()` with the network replaced and
AB_ENDPOINT set to a LOCAL card:

    [framing/raw] answer finish=stop wall=0.0s content=420c
    ...
    wrote ab.json
    endpoint: http://192.168.1.232:8081/v1/chat/completions

That is the whole tail. A sweep that put every run on a card -- the exact
outcome the DISTRIBUTE goal exists to produce -- printed its endpoint and no
count, while the identical sweep on the hosted lane printed a full sentence.
The one case the stack is trying to reach was the one case the summary stayed
silent about.

SCOPE, stated honestly. `_is_hosted` also returns True for None, "" and any URL
naming no host, so an endpoint the summary cannot place would wear the hosted
sentence. That fold is real but NOT reachable through `main()`: `call()` builds
`urllib.request.Request(endpoint)` and a relative or empty endpoint raises
ValueError there, before any summary exists. The fix handles the unknown case
anyway, and this file does not pretend to guard it -- an arm that dies in
`Request()` fails for a reason unrelated to the tally and bounds nothing. It was
written, measured to fail that way, and removed.

So the guard here is the REACHABLE one: a local sweep must say the work went
local, and must not be described as having used no local GPU. Every arm drives
`main()` and reads what it PRINTED. Controls pass on both trees.

    python -m pytest tests/test_nous_ab_lane_unrecorded.py -q
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comms-lane"))

import nous_ab  # noqa: E402
import shape  # noqa: E402


class _Resp(io.BytesIO):
    """What urlopen() returns: a context manager over the body bytes."""

    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


BODY = {
    "choices": [{"finish_reason": "stop",
                 "message": {"content": "a substantive answer " * 20}}],
    "usage": {"completion_tokens": 40},
}

HOSTED = "http://127.0.0.1:8099/v1/chat/completions"
LOCAL = "http://192.168.1.232:8081/v1/chat/completions"

#: The hosted sentence, which must never describe work nobody can place.
HOSTED_SENTENCE = "went to a HOSTED lane"


def _rewrites(task, *_a, **_kw):
    return "REWRITTEN: " + task + " " * 200


def _run(monkeypatch, tmp_path, capfd, endpoint):
    """Drive main() against a fake successful call on `endpoint`."""

    def fake_urlopen(_req, timeout=None):
        return _Resp(BODY)

    monkeypatch.setattr(nous_ab.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shape, "shape", _rewrites)
    monkeypatch.setattr(shape, "shape_with_usage",
                        lambda task, *_a, **_kw: (shape.shape(task, *_a, **_kw), {}))
    monkeypatch.setattr(nous_ab, "OUT_DIR", tmp_path)
    monkeypatch.setattr(nous_ab, "ENDPOINT", endpoint)
    out = tmp_path / "ab.json"
    monkeypatch.setattr(sys, "argv", ["nous_ab.py", "--out", str(out)])
    assert nous_ab.main() == 0
    return capfd.readouterr().out


# ------------------------------------------------------------------ guards ---
# Each of these reads on the pre-fix tree as the wrong answer, not as a missing
# symbol: "" and "v1/chat/completions" both printed the full hosted sentence.

def test_a_local_sweep_says_the_work_went_local(monkeypatch, tmp_path, capfd):
    """The mirror hole: a fully local sweep printed nothing about its lane.

    Pre-fix both branches required `hosted` to be non-zero, so the local case had
    no sentence at all. A reader saw `endpoint: http://192.168.1.232:8081/...`
    and no count, which is the same silence the commit before this one removed.
    """
    printed = _run(monkeypatch, tmp_path, capfd, LOCAL)
    lowered = printed.lower()
    assert "local lane" in lowered, (
        f"a fully local sweep never said the work went to a card:\n{printed}")
    assert "0 local" not in lowered, (
        f"a local-card sweep was reported as having used no local GPU:\n{printed}")


# ---------------------------------------------------------------- controls ---
# These pass on the pre-fix tree as well. A control failing pre-fix would be a
# guard mislabelled; a control passing only post-fix would bound nothing.

def test_control_a_hosted_sweep_still_says_hosted(monkeypatch, tmp_path, capfd):
    printed = _run(monkeypatch, tmp_path, capfd, HOSTED)
    lowered = printed.lower()
    assert HOSTED_SENTENCE in printed, printed
    assert "0 local" in lowered, printed


def test_control_the_summary_still_names_the_model_and_endpoint(
        monkeypatch, tmp_path, capfd):
    printed = _run(monkeypatch, tmp_path, capfd, HOSTED)
    assert nous_ab.MODEL in printed, printed
    assert "SUMMARY" in printed, printed
    assert HOSTED in printed, printed
    assert "wrote" in printed, printed


def test_control_the_run_file_is_still_written_with_its_endpoint(
        monkeypatch, tmp_path, capfd):
    _run(monkeypatch, tmp_path, capfd, LOCAL)
    doc = json.loads((tmp_path / "ab.json").read_text(encoding="utf-8"))
    assert doc["endpoint"] == LOCAL
    assert doc["model"] == nous_ab.MODEL
    assert doc["runs"], "no runs recorded"
