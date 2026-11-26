#!/usr/bin/env python
"""bench_local.py — measure our own model lanes and write a receipt.

Two standard tools answer two different questions, and this script does the
first one because it needs nothing installed:

  SPEED      llama-bench (llama.cpp) or, with zero install, the server's own
             timings. Ollama reports eval_count/eval_duration (decode),
             prompt_eval_count/prompt_eval_duration (prefill) and size_vram
             (did the model actually reach the GPU) on every call. That is
             exact, not a wall clock, and it is what this script reads.
  CAPABILITY lm-evaluation-harness (EleutherAI) — MMLU, GSM8K, ARC, HellaSwag,
             plus HumanEval/MBPP for code. Not installed here; it needs
             transformers+datasets and is a separate, much longer run.

Every number below is measured on this machine and carries the model id, the
quant-independent size, and the VRAM residency, because a tok/s figure with no
residency is the trap this fleet keeps falling into: a model served from system
RAM and one on the card differ by an order of magnitude and read identically in
a chat answer.

Usage:  python bench_local.py [--reps 3] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.error
import urllib.request

# The lanes, as probed. `kind` decides how the rate is obtained.
LANES = [
    # name,          base url,                        model,            kind
    ("nous",         "http://127.0.0.1:8099",         "upstage/solar-mini-4-5", "openai"),
    ("l5090-ollama", "http://192.168.1.232:11434",    "qwen2.5:7b",     "ollama"),
    ("l3090-ollama", "http://192.168.1.124:11434",    "qwen3.5:27b",    "ollama"),
]

# One deterministic prompt. Short, so prefill is dominated by the model, and
# long output, so decode noise averages out.
PROMPT = ("Write exactly ten numbered lines. Each line must be a single short "
          "sentence about hash functions. Do not add any other text.")
NUM_PREDICT = 128


def _post(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def ollama_once(base, model):
    """Exact figures from the server, plus whether the model reached the GPU."""
    url = base + "/api/generate"
    d = _post(url, {"model": model, "prompt": PROMPT, "stream": False,
                    "options": {"num_predict": NUM_PREDICT,
                                "temperature": 0, "seed": 42}})
    dec_n = d.get("eval_count") or 0
    dec_s = (d.get("eval_duration") or 0) / 1e9
    pre_n = d.get("prompt_eval_count") or 0
    pre_s = (d.get("prompt_eval_duration") or 0) / 1e9
    return {
        "decode_tok_s": dec_n / dec_s if dec_s else None,
        "prefill_tok_s": pre_n / pre_s if pre_s else None,
        "load_s": (d.get("load_duration") or 0) / 1e9,
        "decode_tokens": dec_n,
        "prefill_tokens": pre_n,
    }


def ollama_residency(base):
    """size_vram / size per loaded model: 0 means it never left system RAM.

    A field the server did not send is None here, never 0.  The whole point of
    this column is that a model in system RAM and a model on the card read the
    same in a chat answer; rendering an ABSENT ``size`` as ``0%`` invents the
    CPU-resident verdict instead of reporting it, which is the same trap one
    level down.  ``None`` for a failed probe, ``[]`` for a server that answered
    and holds nothing -- those are different facts and the caller keeps them
    apart with ``residency_state``.
    """
    try:
        d = _post(base + "/api/ps", {}, timeout=20)
    except Exception:                              # noqa: BLE001
        return None
    rows = []
    for m in d.get("models", []):
        total = m.get("size")
        vram = m.get("size_vram")
        rows.append({
            "name": m.get("name"),
            "size_reported": total is not None,
            "total_gb": (total / 1e9) if total is not None else None,
            "vram_gb": (vram / 1e9) if vram is not None else None,
            "pct_on_gpu": (100.0 * vram / total)
                          if (total is not None and total and vram is not None)
                          else None,
        })
    return rows


def openai_once(base, model):
    """No server timings, so the rate is a wall clock over usage tokens.

    Stated as the weaker measurement it is: it includes network and scheduling
    overhead the ollama path excludes.
    """
    import time
    url = base + "/v1/chat/completions"
    started = time.time()
    d = _post(url, {"model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": NUM_PREDICT, "temperature": 0,
                    "stream": False})
    elapsed = time.time() - started
    usage = d.get("usage") or {}
    dec_n = usage.get("completion_tokens") or 0
    return {
        "decode_tok_s": dec_n / elapsed if elapsed else None,
        "prefill_tok_s": None,
        "load_s": None,
        "decode_tokens": dec_n,
        "prefill_tokens": usage.get("prompt_tokens"),
        "wall_s": elapsed,
        "cost": usage.get("cost"),
    }


def measure(lane, reps):
    name, base, model, kind = lane
    row = {"lane": name, "model": model, "kind": kind}
    if kind == "ollama":
        res = ollama_residency(base)
        row["residency"] = res
        # Three states, not two: the probe failed, the probe answered and the
        # server holds nothing, the probe answered with models loaded.
        row["residency_state"] = ("unprobed" if res is None
                                  else ("loaded" if res else "empty"))
    samples, error = [], None
    for _ in range(reps):
        try:
            if kind == "ollama":
                samples.append(ollama_once(base, model))
            else:
                samples.append(openai_once(base, model))
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"
            break
        except Exception as exc:                   # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            break
    if error:
        row["error"] = error
        return row
    dec = [s["decode_tok_s"] for s in samples if s.get("decode_tok_s")]
    pre = [s["prefill_tok_s"] for s in samples if s.get("prefill_tok_s")]
    if dec:
        row["decode_tok_s"] = statistics.median(dec)
        row["decode_min"] = min(dec)
        row["decode_max"] = max(dec)
        # Drift is the spread across reps; an effect the size of the drift is
        # not a measurement.
        row["drift_pct"] = (100.0 * (max(dec) - min(dec)) / statistics.median(dec)
                            if statistics.median(dec) else None)
    if pre:
        row["prefill_tok_s"] = statistics.median(pre)
    row["reps"] = len(samples)
    return row


def render(rows):
    out = []
    out.append("| lane | model | decode tok/s | drift | prefill tok/s | on GPU |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['lane']} | {r['model']} | **FAILED** | — | — | "
                       f"{r['error'][:60]} |")
            continue
        state = r.get("residency_state")
        res = r.get("residency")
        if state == "unprobed":
            gpu = "unmeasured (/api/ps failed)"
        elif res is None and not state:
            # Legacy row with no state: unknown, not measured-at-zero.
            gpu = "unmeasured"
        elif not res:
            gpu = "none loaded"
        else:
            hit = [m for m in res if m["name"] == r["model"]]
            if not hit:
                gpu = "not loaded"
            elif hit[0].get("pct_on_gpu") is None:
                # size or size_vram was never sent: no percentage exists.
                gpu = "size not reported"
            else:
                gpu = f"{hit[0]['pct_on_gpu']:.0f}% ({hit[0]['vram_gb']:.2f} GB)"
        drift = (f"{r['drift_pct']:.1f}%" if r.get("drift_pct") is not None
                 else "—")
        # `is not None`, because a measured 0.0 tok/s is a measurement and
        # must not render as the same dash a missing sample renders as.
        dec = (f"{r['decode_tok_s']:.1f}" if r.get("decode_tok_s") is not None
               else "—")
        pre = (f"{r['prefill_tok_s']:.0f}" if r.get("prefill_tok_s") is not None
               else "—")
        out.append(f"| {r['lane']} | {r['model']} | {dec} | {drift} | {pre} | {gpu} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = [measure(lane, args.reps) for lane in LANES]
    table = render(rows)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# Local lane bench\n\n")
            fh.write(f"reps={args.reps}, num_predict={NUM_PREDICT}, "
                     "temperature=0, seed=42\n\n")
            fh.write(table + "\n\n")
            fh.write("```json\n" + json.dumps(rows, indent=2) + "\n```\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
