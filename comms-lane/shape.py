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
import re
import sys
import urllib.request

OLLAMA = "http://192.168.1.124:11434"
MODEL = "defiant9b-abliterated:256k"
TIMEOUT = 180

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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return _strip_thinking(data.get("response", ""))


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
    except Exception as e:
        print(f"SHAPE_FAIL: {e}", file=sys.stderr)
        sys.exit(1)
