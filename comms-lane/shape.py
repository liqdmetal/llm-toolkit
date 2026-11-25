"""shape.py — prompt-shaping layer for the frontier lane (comms lane).

Route: task prompt -> defiant9b-abliterated (3090 ollama, the Rosetta/defiant
box at 192.168.1.124:11434) -> shaped prompt -> codebuff
(claude-haiku-5.5 / gpt-6-luna-pro).

What it is: prompt SHAPING, not jailbreaking. The frontier model still
refuses when its own guardrails trip; we just stop handing it the tripwire
(emotional framing, adversarial phrasing, giant undivided asks). We also gain
the decompose step: a giant ask becomes numbered sub-asks.

Stdlib only. Usage:
    python shape.py "task text"                # shaped prompt on stdout
    python shape.py --system-file my.txt "t"
Exits 0 with the shaped prompt, 1 on shaper failure (caller should fall
through to the raw prompt — never let shaping be a hard dependency).
"""
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

OLLAMA = os.environ.get("SHAPE_OLLAMA", "http://192.168.1.124:11434")
MODEL = os.environ.get("SHAPE_MODEL", "defiant9b-abliterated:256k")
# Generation timeout only. It is generous because the shaper may emit a long
# thinking preamble before the rewritten prompt.
TIMEOUT = int(os.environ.get("SHAPE_TIMEOUT", "180"))
# Reachability probe. Deliberately small: this answers "is the box there at
# all", which must not cost anywhere near a generation timeout.
PROBE_TIMEOUT = float(os.environ.get("SHAPE_PROBE_TIMEOUT", "2.0"))


class ShaperUnavailable(RuntimeError):
    """The shaper box cannot be reached, so shaping must be skipped.

    Distinct from a shaping *failure*: this is raised BEFORE any generation is
    attempted, so a caller falls through to the raw prompt immediately instead
    of blocking for TIMEOUT seconds to learn the same thing.
    """


# A model that is LISTED but cannot be LOADED is not a shaping failure -- it is
# the box being unable to serve the model right now, which ROUTING.md rule 1
# says must fall through to the raw prompt. Ollama reports it as a generic
# HTTP 500 whose body carries the backend's own message, so the body is the
# only place the distinction exists. MEASURED 2026-11-28 against the real 3090
# (192.168.1.124:11434) with defiant9b-abliterated:256k listed and Inkling
# holding the card:
#
#   HTTP 500 {"error": "llama-server process has terminated: exit status 1:
#                     ggml_backend_cuda_device_get_memory: cudaMemGetInfo
#                     failed (out of memory) ... cudaMalloc failed: out of
#                     memory ... failed to allocate CUDA0 buffer of size
#                     52690944 ... failed to initialize the context"}
#
# Through the CLI that read `SHAPE_FAIL: HTTP Error 500: Internal Server
# Error` where the contract calls for `SHAPE_SKIP: ...`.
_LOAD_FAILURE_MARKERS = (
    "out of memory",
    "failed to allocate",
    "llama-server process has terminated",
    "failed to load model",
    "unable to load model",
)


def _is_model_load_failure(body: str) -> bool:
    """True only for a body that says the model could not be brought up.

    Deliberately narrow: an unrelated 500 must stay a raised error rather than
    being relabelled a skip, or every server bug becomes silent capacity loss.
    """
    low = body.lower()
    return any(m in low for m in _LOAD_FAILURE_MARKERS)



def _host_port(ollama: str | None = None) -> tuple[str, int]:
    u = urllib.parse.urlsplit(ollama or OLLAMA)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    return host, port


def _installed_models(ollama: str | None, timeout: float) -> set[str] | None:
    """The model names the shaper host lists, or None if it cannot be read.

    None is NOT an empty set. A listing endpoint that errors says nothing about
    which models exist, and reading it as "none installed" would refuse every
    model on a box whose /api/tags is merely unhealthy -- turning a soft
    dependency into a guaranteed skip.

    /api/tags is a listing endpoint: it reads state and loads nothing, so asking
    cannot evict the model this is meant to protect.
    """
    base = (ollama or OLLAMA).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as r:
            doc = json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError):
        return None
    return {m.get("name") for m in doc.get("models", []) if m.get("name")}


def preflight(ollama: str | None = None, timeout: float = PROBE_TIMEOUT,
              model: str | None = None) -> None:
    """Raise ShaperUnavailable unless the box is reachable AND carries the model.

    Two questions, because only the first used to be asked and the second is the
    one a config typo breaks:

    1. Is the host listening? A TCP connect answers that in milliseconds, where
       a generation attempt would block for TIMEOUT to learn the same thing.
    2. Does it LIST the configured model? A name that is not installed gets a
       verbatim HTTP 404 from Ollama's generation endpoint -- MEASURED at 11 ms,
       so this is not a timeout problem, it is a CONTRACT problem. ROUTING.md
       rule 1 tells callers to catch ShaperUnavailable and fall back to their
       own prompt, and an escaping HTTPError is an exception type no caller is
       told to handle. Shaping then hard-blocks the work it exists to help.

    A listing that FAILS cannot answer question 2, so it does not refuse: only a
    listing that ANSWERS and omits the model does.
    """
    host, port = _host_port(ollama)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        raise ShaperUnavailable(
            f"{host}:{port} unreachable ({e.__class__.__name__})") from e

    wanted = model or MODEL
    installed = _installed_models(ollama, timeout)
    if installed is not None and wanted not in installed:
        near = sorted(n for n in installed
                      if wanted.split(":")[0].split("/")[-1] in n)[:3]
        hint = "; nearest on that host: " + ", ".join(sorted(near)) if near else ""
        raise ShaperUnavailable(
            f"model {wanted!r} is not installed on {host}:{port} "
            f"({len(installed)} model(s) listed){hint}")


def enabled() -> bool:
    """Shaping is opt-OUT, so a reserved or busy box can be declared off."""
    return os.environ.get("SHAPE_DISABLE", "").strip().lower() not in ("1", "true", "yes", "on")

DEFAULT_SYSTEM = """You are a prompt engineer for a technical research fleet.

Rewrite the user's task into a neutral, factual, decomposable instruction
that a frontier model will answer fully and precisely.

Rules:
- Keep every technical requirement, constraint, path, and variable intact.
- Replace emotionally loaded or adversarial framing with plain statements.
- Split a giant ask into numbered sub-asks.
- Prefer concrete verbs: analyze, identify, verify, implement, report,
  enumerate, compare.
- Do not add or remove constraints. Do not ask the model to ignore rules.
- Do not mention 'jailbreak', 'uncensored', 'guardrail', or any policy topic
  unless the original task requires the model to answer a policy question.
- Output ONLY the rewritten prompt. No preamble, no explanation."""


def _strip_thinking(text: str) -> str:
    """Remove the thinking preamble the model emits before its answer."""
    # cut at common markers; keep the final rewritten prompt
    cut = None
    for marker in ("Final Answer:", "FINAL ANSWER:", "Rewritten prompt:",
                   "Here is the rewritten", "Output:"):
        i = text.rfind(marker)
        if i != -1:
            cut = i + len(marker)
            break
    if cut is None:
        # no marker: drop an explicit "Thinking Process" block if present
        i = text.find("Thinking Process:")
        if i != -1:
            text = text[i + len("Thinking Process:"):]
        # then keep the last paragraph block after the thinking enumeration
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if parts and re.match(r"^\d+\.", parts[0]):
            return " ".join(parts[1:]) if len(parts) > 1 else parts[-1]
        return text.strip()
    return text[cut:].strip()


def shape(task: str, system: str = DEFAULT_SYSTEM, timeout: int = TIMEOUT) -> str:
    """Shape a task and return ONLY the rewritten prompt.

    Kept as the primary entry point because ROUTING.md makes shaping a SOFT
    dependency and every existing caller expects a string. Callers that need
    to account for what shaping COSTS use shape_with_usage().
    """
    text, _usage = shape_with_usage(task, system, timeout)
    return text


def shape_with_usage(task: str, system: str = DEFAULT_SYSTEM,
                     timeout: int = TIMEOUT) -> tuple[str, dict]:
    """Shape a task, returning the rewrite AND what producing it cost.

    WHY THIS EXISTS
    ---------------
    `shape()` returned a string and threw away the rest of the ollama response.
    That response carries `prompt_eval_count`, `eval_count` and the durations —
    the actual token cost of shaping — so NO caller could report what the
    shaping step spends. "Shaping saves frontier tokens" is then an unmeasured
    claim, which is precisely what this stack exists to refuse.

    MEASURED 2026-11-24 against 192.168.1.124:11434, defiant9b-abliterated:256k,
    this file's own DEFAULT_SYSTEM and the nous_ab "framing" task:

        prompt_eval_count   237 tokens IN   (system prompt + task)
        eval_count          125 tokens OUT  (the rewrite)
        load_ms            3509.4
        eval_ms            1465.4
        wall_s                5.08
        num_predict cap       4096  ->  cap utilisation 3.1%
        task chars in          199
        shaped chars out       623  ->  a 3.13x expansion

    The rewrite is 3.13x the input's character length, so shaping is not
    free on the prompt side either: it ADDS prompt tokens to save completion
    tokens. Neither side of that trade was measurable before this function.

    The usage dict is always present and always states its own provenance:
    keys are absent when the server did not report them, never zero-filled —
    an absent count is not a measured zero.
    """
    if not enabled():
        raise ShaperUnavailable("shaping disabled by SHAPE_DISABLE")
    preflight()
    payload = {
        "model": MODEL,
        "prompt": task,
        "system": system,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # The installed set can change between the preflight and the
            # generation, so the same contract is enforced here too: the caller
            # sees ShaperUnavailable, never a raw HTTPError.
            _h, _p = _host_port()
            raise ShaperUnavailable(
                f"model {MODEL!r} is not installed on {_h}:{_p} "
                f"(HTTP 404 at generation time)") from e
        # A listed model that cannot be brought up (typically the card is full)
        # is the same contract: skip to the raw prompt, do not hard-block.
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:                                   # noqa: BLE001
            body = ""
        if _is_model_load_failure(body):
            _h, _p = _host_port()
            raise ShaperUnavailable(
                f"model {MODEL!r} is listed on {_h}:{_p} but could not be "
                f"loaded (HTTP {e.code}); that box has no room for it right "
                f"now") from e
        raise
    raw = data.get("response", "")
    shaped = _strip_thinking(raw)

    # Only report what the server actually said. `or 0` would turn an absent
    # field into a measured zero, which is the defect class this repo keeps
    # meeting: a count that is present because someone defaulted it.
    usage = {"model": data.get("model") or MODEL, "basis": "measured"}
    for src, dst in (("prompt_eval_count", "prompt_tokens"),
                     ("eval_count", "completion_tokens"),
                     ("load_duration", "load_ns"),
                     ("prompt_eval_duration", "prompt_eval_ns"),
                     ("eval_duration", "eval_ns"),
                     ("total_duration", "total_ns")):
        if src in data:
            usage[dst] = data[src]
    # What the rewrite does to the prompt it replaces — the other half of the
    # trade, and the half nobody could see.
    usage["task_chars"] = len(task)
    usage["shaped_chars"] = len(shaped)
    if task:
        usage["expansion_x"] = round(len(shaped) / len(task), 2)
    # A shaper that produced nothing still cost tokens. Say so rather than
    # returning an empty string that reads like a free no-op.
    usage["produced_text"] = bool(shaped.strip())
    return shaped, usage


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: shape.py [--system-file FILE] <task>", file=sys.stderr)
        sys.exit(2)
    system = DEFAULT_SYSTEM
    if "--system-file" in sys.argv:
        i = sys.argv.index("--system-file")
        with open(sys.argv[i + 1], encoding="utf-8") as f:
            system = f.read()
        task = sys.argv[i + 2]
    else:
        task = sys.argv[1]
    try:
        print(shape(task, system))
    except ShaperUnavailable as e:
        # Soft dependency: the caller uses its original prompt. Reported on a
        # distinct marker so a skip is never mistaken for a shaping failure.
        print(f"SHAPE_SKIP: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"SHAPE_FAIL: {e}", file=sys.stderr)
        sys.exit(1)
