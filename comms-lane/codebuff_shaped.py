"""codebuff_shaped.py — run codebuff with the shaping layer in front.

Route: task prompt -> shape.py (defiant9b-abliterated on the 3090) ->
codebuff (frontier model, e.g. anthropic/claude-haiku-5.5,
openai/gpt-6-luna-pro).

What it is: prompt SHAPING, not jailbreaking. The frontier model still
refuses when its own guardrails trip; shaping only stops us handing it the
tripwire (emotional framing, adversarial phrasing, giant undivided asks).
On shaper failure we fall through to the raw task — shaping is never a hard
dependency.

Usage:
    python codebuff_shaped.py "<task>" --cwd <dir> [--model <id>] [--dry-run]

--dry-run prints the exact codebuff command without running it.
codebuff output (stdout/stderr/exit code) passes through.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shape  # noqa: E402

CODEFREEBUFF = os.environ.get(
    "CODEFREEBUFF_BIN",
    str(Path.home() / "AppData" / "Local" / "hermes" / "bin" / "codebuff"),
)
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


def main() -> int:
    ap = argparse.ArgumentParser(description="codebuff via shaping layer")
    ap.add_argument("task", help="the task, in the user's words")
    ap.add_argument("--cwd", required=True, help="repo/working dir for codebuff")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="codebuff model id")
    ap.add_argument("--dry-run", action="store_true", help="print, don't run")
    args = ap.parse_args()

    shaped = shape.shape(args.task)
    if not shaped:
        print(f"[codebuff_shaped] shaper returned empty; using raw task",
              file=sys.stderr)
        shaped = args.task

    cmd = [CODEFREEBUFF, shaped, "--cwd", args.cwd, "--model", args.model,
           "--json"]
    print(f"[codebuff_shaped] shaped prompt ({len(shaped)} chars):\n"
          f"{shaped}\n", file=sys.stderr)
    print(f"[codebuff_shaped] command: "
          f"{' '.join(c if ' ' not in c else chr(34)+c+chr(34) for c in cmd)}",
          file=sys.stderr)
    if args.dry_run:
        print("DRY_RUN_OK", file=sys.stderr)
        return 0

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
