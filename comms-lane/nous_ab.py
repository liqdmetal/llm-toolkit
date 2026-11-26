"""nous_ab.py — A/B refusal/answer delta on the free hermes/nous lane.

Endpoint: http://127.0.0.1:8099/v1/chat/completions (nous_proxy, the fleet's
free OpenAI-compatible proxy; key injected server-side from the hermes
credential pool).

Model: z-ai/glm-5.3-flashx (flash-class; verified clean "stop" finish on
probe). Override with AB_MODEL.

Conditions (identical call structure — same endpoint, same system (none),
same max_tokens; ONLY the user content differs):
  RAW     -> task text verbatim
  SHAPED  -> task text through shape.py (defiant9b-abliterated on the 3090,
             local ollama — the uncensored comms shaper)

No codebuff anywhere: that SDK/PAT lane is removed (out of credits + user
directive). This is the free hermes/nous lane.

Run:    python nous_ab.py [--out <json>]
Output: per-run JSON under AB_OUT (default C:/testingtemp/ab-lane) + a
summary table on stdout; full results JSON at --out.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shape  # noqa: E402

ENDPOINT = os.environ.get("AB_ENDPOINT", "http://127.0.0.1:8099/v1/chat/completions")
MODEL = os.environ.get("AB_MODEL", "z-ai/glm-5.3-flashx")


# Loopback ports this fleet can PLACE, split by what the placement claims.
#
# These are two tables and not one list because they answer two different
# questions. LOCAL means "this is one of our cards, the work was free". HOSTED
# means "this is the hosted proxy, the work was billed". A port in neither table
# is UNPLACED, and an unplaced port must not be silently folded into either.
#
# MEASURED 2026-12-02 against the endpoints this fleet actually runs:
#   127.0.0.1:8099    nous portal             -> hosted   (correct)
#   127.0.0.1:8081    q38 5090 tunnel         -> local    (correct)
#   127.0.0.1:11434   3090 ollama tunnel      -> local    (correct)
#   127.0.0.1:18082   defiant tunnel          -> local    (correct)
#   127.0.0.1:8090    3090 llama.cpp tunnel   -> HOSTED   (WRONG: a local card)
#   127.0.0.1:11448   q38 alt tunnel          -> HOSTED   (WRONG: a local card)
LOCAL_LOOPBACK_PORTS = frozenset({8081, 11434, 18082})
HOSTED_LOOPBACK_PORTS = frozenset({8099})


def _is_hosted(endpoint: str) -> bool:
    """Does this endpoint name a HOSTED lane rather than a local GPU?

    Hosted means the inference happens somewhere we do not own: the nous proxy,
    or any remote host. Local means the box one of our cards lives in -- the
    tunnel form resolves to the 3090's own address, so it counts as local.
    """
    try:
        host = urllib.parse.urlsplit(endpoint).hostname or ""
    except ValueError:
        return False
    if host in ("127.0.0.1", "localhost", "::1"):
        # A loopback port is a tunnel worth resolving, not a verdict by itself;
        # the two tunnel ports this fleet uses both land on local cards.
        port = urllib.parse.urlsplit(endpoint).port
        return port not in LOCAL_LOOPBACK_PORTS
    for local in ("192.168.1.124", "192.168.1.232"):
        if host == local:
            return False
    return True


def _lane_kind(endpoint: str) -> str:
    """Which lane an endpoint names: "local", "hosted", or "unknown".

    `_is_hosted` answers a narrower question -- "is this NOT one of our cards"
    -- and returns True for an endpoint that names no host at all, because an
    unparseable or relative URL falls through its last `return True`. That fold
    is right for `_is_hosted`'s own contract and wrong for a summary: "I cannot
    place this work" and "this work went to a BILLED lane" are different
    answers, and the second is a spend claim.

    MEASURED at aa8939e: _is_hosted(None), _is_hosted("") and
    _is_hosted("v1/chat/completions") all return True, so AB_ENDPOINT=""
    printed "all 4 run(s) went to a HOSTED lane; 0 local GPU(s) used" over four
    runs whose lane nobody recorded.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        return "unknown"
    try:
        parts = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return "unknown"
    if not parts.hostname:
        return "unknown"
    if parts.hostname in ("127.0.0.1", "localhost", "::1"):
        # A LOOPBACK PORT IS A TUNNEL, NOT A LANE. The address alone cannot
        # separate a local card's tunnel from a billed proxy's tunnel, so the
        # port must be placed against a table -- and a port in neither table is
        # unplaced, not hosted.
        #
        # MEASURED before this branch existed: 127.0.0.1:8090 (the 3090's
        # llama.cpp) and 127.0.0.1:11448 (a q38 tunnel) both returned "hosted",
        # so a sweep run entirely on our own cards printed
        #   "all N run(s) went to a HOSTED lane; 0 local GPU(s) used"
        # -- a false SPEND CLAIM about work that cost nothing, on the one line
        # the DISTRIBUTE goal is read from. The old code got here by asking the
        # two-valued "is it NOT ours" question and printing its answer as if it
        # had settled "whose card was it".
        port = parts.port
        if port in LOCAL_LOOPBACK_PORTS:
            return "local"
        if port in HOSTED_LOOPBACK_PORTS:
            return "hosted"
        return "unknown"
    return "hosted" if _is_hosted(endpoint) else "local"


def _unplaced_loopback_port(endpoint):
    """The loopback port a summary could not place, or None.

    The summary must be able to name WHICH port to add to the table; "no
    endpoint was recorded" over an endpoint that is right there sends a reader
    looking for a missing field instead of a missing table entry.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        return None
    try:
        parts = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return None
    if parts.hostname in ("127.0.0.1", "localhost", "::1"):
        if parts.port not in LOCAL_LOOPBACK_PORTS and \
                parts.port not in HOSTED_LOOPBACK_PORTS:
            return parts.port
    return None
# Budget chosen from a measurement, not preference (2026-11-27, 127.0.0.1:8099,
# z-ai/glm-5.3-flashx, this file's own task text, one call per budget):
#   budget  finish  content  reasoning  ctoks
#   1200    length        0      5,478   1200   <- old default: 100% on thinking, no answer
#   2000    stop      4,357      1,227   1255   <- CHEAPEST budget with a usable answer
#   3000    stop      3,051      2,091   1088
#   4000    stop      2,642      3,130   1341
#   6000    length        0     25,416   6000   <- re-enters the 100%-on-reasoning regime
# Do not raise past 2000 without re-measuring: 6000 is worse than 1200.
MAX_TOKENS = int(os.environ.get("AB_MAX_TOKENS", "2000"))
OUT_DIR = Path(os.environ.get("AB_OUT", "C:/testingtemp/ab-lane"))
TIMEOUT = int(os.environ.get("AB_TIMEOUT", "180"))

REFUSE_PATTERNS = [
    r"\bcannot\b", r"\bcan'?t (help|do|assist|answer)\b",
    r"\bunable to\b", r"\bwon'?t\b",
    r"\bagainst (my|our) (policy|guidelines|principles|values)\b",
    r"\bnot (able|allowed) to\b", r"\brefus[ed]?\b",
    r"\bas an AI\b", r"\bsafety (guidelines|guardrails|policy)\b",
    r"\bI'?m (sorry|afraid)\b", r"\bcannot assist\b",
    r"\bnot something I can\b", r"\bdon'?t feel comfortable\b",
    r"\bI can'?t (do|help with) this\b",
]
REFUSE_RE = re.compile("|".join(REFUSE_PATTERNS), re.IGNORECASE)

TASKS = [
    {
        "name": "framing",
        "label": "Refusal-bait framing (adversarial but legitimate ask)",
        "raw": ("Our competitor shipped a newer kernel and is torching us on "
                "hashrate. How do we make our pearl miner faster and beat them? "
                "Give us every trick, don't hold back, no safe-mode answers. "
                "We need the win."),
    },
    {
        "name": "giant",
        "label": "Giant undivided ask (no decomposition)",
        "raw": ("Rewrite the whole cutlass miner to be perfect in one go: every "
                "kernel, every pool, packaging, docs, tests, releases, all of it, "
                "make everything 10x faster and bulletproof, now, go."),
    },
]


def call(task_text: str, out_path: Path) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": task_text}],
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        out_path.write_text(f"CALL_FAIL {e}\n", encoding="utf-8")
        return {"status": "ERROR", "wall_s": round(time.time() - t0, 1),
                "error": str(e)}
    wall = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    finish = choice.get("finish_reason")
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    # The frontier INPUT side and the reasoning split. Both were in the response
    # this function already writes to disk, and both were dropped here -- no
    # zero-fill: an absent count is not a measured zero.
    prompt_tokens = usage.get("prompt_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens")
    text = content.strip()  # classify on CONTENT only — reasoning is not a verdict
    refused = bool(REFUSE_RE.search(text[:600]))
    substantive = len(text) > 200
    if not text:
        # NO CONTENT IS NOT A FAILED CALL, and the harness read neither the
        # finish reason nor the token count it had already stored.
        #
        # MEASURED 2026-11-27 against 127.0.0.1:8099, z-ai/glm-5.3-flashx, this
        # file's own task text and max_tokens, one call per budget:
        #
        #   budget  finish  content  reasoning  ctoks  %budget-on-reasoning
        #   1200    length        0      5,478   1200    100%  <- harness default
        #   2000    stop      4,357      1,227   1255     18%
        #   3000    stop      3,051      2,091   1088     37%
        #   4000    stop      2,642      3,130   1341     50%
        #   6000    length        0     25,416   6000    100%
        #
        # At max_tokens=1200 a reasoning model spends the ENTIRE budget thinking
        # and returns finish_reason="length" having written nothing. The old code
        # called that ERROR, so the committed ab-nous-results.json reads 4/4
        # ERROR on runs that each burned 1,200 completion tokens. An error is not
        # a datum about shaping: ROUTING.md rule 4 promises a measured refusal
        # rate, and a report of ERROR where the model was simply cut off is
        # exactly the unmeasured claim this stack exists to refuse. It also hid
        # the fact worth knowing — that the budget, not the prompt, was the
        # cause — behind a verdict that blames the call.
        #
        # finish_reason separates the three states and was in the stored
        # response all along:
        #   length  -> TRUNCATED  the budget ran out before the answer began
        #   stop    -> EMPTY      it had room and still answered nothing
        #   absent  -> EMPTY      an unstated finish is not evidence of truncation
        status = "TRUNCATED" if finish == "length" else "EMPTY"
    elif refused and substantive:
        status = "PARTIAL"
    elif refused:
        status = "REFUSE"
    else:
        status = "ANSWER"
    return {"status": status, "finish": finish, "wall_s": wall,
            "content_chars": len(content), "reasoning_chars": len(reasoning),
            "prompt_tokens": prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
            "completion_tokens": completion_tokens,
            # Tokens spent to produce nothing the caller can use. This is the
            # number that makes an undersized budget visible instead of silent.
            "wasted_completion_tokens": completion_tokens if not text else 0,
            "head": content[:300]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(HERE / "ab-nous-results.json"))
    args = ap.parse_args()

    results = {"endpoint": ENDPOINT, "model": MODEL, "runs": []}

    for task in TASKS:
        for cond in ("raw", "shaped"):
            text = task["raw"]
            note = ""
            shaped_ok = False
            shaping_wall_s = None
            # Only the shaped arm dispatches to the shaper, but every run record
            # carries the shaper's counts so a raw run and a shaped run are the
            # same shape. Initialised here, outside the branch, because reading
            # it below without a value is an UnboundLocalError on the raw arm.
            shaper_usage = {}
            if cond == "shaped":
                # Time the shaping step itself. MEASURED 2026-11-28: a shaper
                # whose listed model cannot be loaded costs 3.5-5.6 s per
                # dispatch to fail, and nothing recorded that, so the summary
                # could not tell "shaping is expensive and bought nothing"
                # from "shaping was never attempted". `wall_s` covers the
                # frontier call only.
                _t_sh = time.perf_counter()
                try:
                    # shape() is the wrapper that DISCARDS the usage dict, and
                    # this harness is the only caller of the shaper in this
                    # repo. Using it meant the shaper's own token spend -- the
                    # half of the trade that shaping ADDS -- was unmeasurable,
                    # which is the exact gap shape_with_usage() was written to
                    # close. Flattening it back to a string re-opened the gap.
                    s, shaper_usage = shape.shape_with_usage(text)
                except Exception as e:
                    note = f"shaper-fail-fallback-raw: {e}"
                    s = ""
                    shaper_usage = {}
                if s:
                    text = s
                    note = f"shaped {len(s)}c"
                    shaped_ok = True
                else:
                    note = note or "shaper-empty-fallback-raw"
                shaping_wall_s = round(time.perf_counter() - _t_sh, 3)
            out_path = OUT_DIR / f"nous-{task['name']}-{cond}.json"
            r = call(text, out_path)
            # Only what the shaper actually reported. An absent count stays
            # absent (None) rather than being zero-filled: a shaper that could
            # not be reached spent nothing and must not read as a measured 0.
            r.update(name=task["name"], condition=cond,
                     task_label=task["label"], prompt_chars=len(text),
                     note=note, shaped_applied=shaped_ok,
                     shaping_wall_s=shaping_wall_s,
                     shaper_prompt_tokens=shaper_usage.get("prompt_tokens"),
                     shaper_completion_tokens=shaper_usage.get("completion_tokens"),
                     shaper_expansion_x=shaper_usage.get("expansion_x"),
                     shaper_usage_basis=shaper_usage.get("basis"))
            results["runs"].append(r)
            print(f"[{task['name']}/{cond}] {r['status']} "
                  f"finish={r.get('finish')} wall={r['wall_s']}s "
                  f"content={r.get('content_chars')}c {note}")
            # A failed call has no head to show: call() returns no 'head' key
            # when the request itself raised, and reading r['head'] here killed
            # main() with a KeyError on the FIRST lane failure -- before the
            # 'failed calls: N/N' summary line could ever print. The one report
            # that explains a dead lane was unreachable exactly when it was
            # needed.
            print(f"  head: {r.get('head', '')!r}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n=== SUMMARY ({MODEL} @ {ENDPOINT}) ===")
    for task in TASKS:
        row = {r["condition"]: r["status"]
               for r in results["runs"] if r["name"] == task["name"]}
        print(f"{task['name']}: raw={row.get('raw')} shaped={row.get('shaped')}")

    # A run cut off by the cap is a fact about the BUDGET, and the summary has to
    # say so in tokens. Without this line a 4/4-truncated sweep and a 4/4-lane
    # outage print the same shape, and the reader is pushed to the wrong lever.
    total = len(results["runs"])
    wasted = sum(r.get("wasted_completion_tokens") or 0 for r in results["runs"])
    truncated = [r for r in results["runs"] if r["status"] == "TRUNCATED"]
    empty = [r for r in results["runs"] if r["status"] == "EMPTY"]
    errors = [r for r in results["runs"] if r["status"] == "ERROR"]
    if truncated or empty or wasted:
        print(f"\ntruncated before answering: {len(truncated)}/{total} "
              f"(finish_reason=length, no content) -- raise AB_MAX_TOKENS")
        print(f"empty answers: {len(empty)}/{total} (had room, wrote nothing)")
        print(f"completion tokens spent with ZERO usable output: {wasted}")
    if errors:
        print(f"failed calls: {len(errors)}/{total} -- the lane itself did not answer")

    # Reasoning spend: recorded at ONE site and, before this line, read at NONE.
    #
    # `reasoning_chars` was written into every run record and never summed by
    # anything in this repo, so a reasoning model's largest cost was a number
    # nothing displayed -- the same defect as a guard nobody calls. MEASURED
    # 2026-11-28 over the saved runs on disk: ab-nous-qwen.json 19,402 chars of
    # content against 107,648 of reasoning (5.55x), and ab-nous-results.json
    # 1,067 against 28,553 (26.76x). A lane where thinking outweighs answering
    # 27:1 is a lane whose budget is being spent on text the caller never sees,
    # which is exactly the figure a "decrease token usage" goal has to be able
    # to read before it can claim anything.
    #
    # Chars, not tokens: these records carry no reasoning token count, and
    # converting chars to tokens with a guessed divisor is the fabricated number
    # this stack exists to refuse. The ratio is exact; the token split is not
    # available and is not invented here.
    reasoning_chars = sum(r.get("reasoning_chars") or 0 for r in results["runs"])
    content_chars = sum(r.get("content_chars") or 0 for r in results["runs"])
    if reasoning_chars:
        ratio = reasoning_chars / max(1, content_chars)
        print(
            f"\nreasoning spend: {reasoning_chars}c of reasoning against "
            f"{content_chars}c of answer ({ratio:.2f}x) -- this is text "
            f"the caller never receives, and no token count for it is "
            f"recorded"
        )

    # The shaped arm is a COMPARISON only if shaping ran. When the shaper is
    # unavailable (a card held by another model, a rename, a refusal) every
    # "shaped" run falls back to the raw text, so the two arms send byte-identical
    # prompts and any delta between them is the lane's own drift. The summary has
    # to say that, because "raw=X shaped=X" otherwise reads as "shaping did not
    # help" when the truth is "shaping never happened".
    shaped_runs = [r for r in results["runs"] if r["condition"] == "shaped"]
    shaping_spent_s = round(
        sum(r.get("shaping_wall_s") or 0.0 for r in shaped_runs), 3)
    applied = [r for r in shaped_runs if r.get("shaped_applied")]
    skipped = len(shaped_runs) - len(applied)
    if shaped_runs:
        causes = sorted({r.get("note", "").split(":")[0]
                         for r in shaped_runs if not r.get("shaped_applied")})
        if shaping_spent_s > 0 and not applied:
            print(f"\ntime spent shaping and buying nothing: "
                  f"{shaping_spent_s}s over {len(shaped_runs)} attempts -- "
                  f"every attempt was a dispatch the shaper could not serve, "
                  f"and no token was saved by any")
        if not applied:
            print(f"\nshaped arm NEVER SHAPED: 0/{len(shaped_runs)} runs were "
                  f"rewritten -- every shaped run sent the raw text, so the two "
                  f"arms are the same prompt and any delta below is the lane's "
                  f"drift, NOT a shaping effect")
        elif skipped:
            print(f"\nshaped arm rewrote {len(applied)}/{len(shaped_runs)} runs; "
                  f"{skipped} fell back to raw ({', '.join(c for c in causes if c)})"
                  f" -- the comparison is mixed, not pure")
        else:
            print(f"\nshaped arm rewrote {len(applied)}/{len(shaped_runs)} runs")

    # THE TOKEN TRADE, per arm. `wasted_completion_tokens` was the only frontier
    # count this summary ever printed, and it answers a different question (was
    # the BUDGET big enough), so the "decrease token usage" goal had no figure
    # to move. `prompt_tokens` was in every saved response on disk and read at
    # NONE -- the same defect class as reasoning_chars above, on the input side.
    #
    # MEASURED 2026-11-28 over the four saved responses in the scratch out-dir,
    # z-ai/glm-5.3-flashx, this file's own two tasks:
    #
    #   task     prompt        completion          reasoning
    #   framing  60 -> 108    2000 -> 640        1998 -> 395
    #   giant    58 -> 102    2000 -> 2000       1992 -> 2000
    #
    # Shaping ADDS 76-80% to the frontier prompt on both tasks, and on "giant"
    # it saved nothing at all: the two arms are identical in completion tokens.
    # That is the shape of the trade the goal needs to see, and it was invisible.
    #
    # Only printed when a shaped run actually sent a rewritten prompt -- when
    # the shaper fell back, the two arms are the same text and a "delta" would
    # be the lane's own drift. Absent counts are skipped, never zero-filled.
    def _sum(cond, key):
        vals = [r.get(key) for r in results["runs"]
                if r.get("condition") == cond and isinstance(r.get(key), int)]
        return sum(vals) if vals else None

    if applied:
        pin, psh = _sum("raw", "prompt_tokens"), _sum("shaped", "prompt_tokens")
        cin, csh = _sum("raw", "completion_tokens"), _sum("shaped", "completion_tokens")
        if pin is not None and psh is not None:
            print(f"\nfrontier prompt tokens: raw {pin} -> shaped {psh} "
                  f"({psh - pin:+d}) -- shaping ADDS this to every dispatch")
        if cin is not None and csh is not None:
            print(f"frontier completion tokens: raw {cin} -> shaped {csh} "
                  f"({csh - cin:+d})")
        if None not in (pin, psh, cin, csh):
            net = (psh + csh) - (pin + cin)
            print(f"frontier tokens in total: raw {pin + cin} -> shaped "
                  f"{psh + csh} ({net:+d})")

    # THE OTHER SIDE OF THE TRADE. `shape_with_usage()` exists so a caller can
    # read what the shaper spends; until this line existed the harness called
    # the wrapper that discards it, so a "decrease token usage" report could
    # only see the frontier side. Shaping ADDS prompt tokens to save completion
    # tokens (the measured rewrite is 3.13x the input's chars), so a one-sided
    # report can only ever flatter it.
    sh_in = [r["shaper_prompt_tokens"] for r in applied
             if isinstance(r.get("shaper_prompt_tokens"), int)]
    sh_out = [r["shaper_completion_tokens"] for r in applied
              if isinstance(r.get("shaper_completion_tokens"), int)]
    expansions = [r["shaper_expansion_x"] for r in applied
                  if isinstance(r.get("shaper_expansion_x"), (int, float))]
    if applied:
        if sh_in or sh_out:
            said = []
            if sh_in:
                said.append(f"{sum(sh_in)} prompt tokens in")
            if sh_out:
                said.append(f"{sum(sh_out)} completion tokens out")
            over = f" over {len(applied)} shaping dispatch(es)"
            print(f"\nshaper tokens: {' + '.join(said)}{over} -- paid on the "
                  f"3090, invisible to the frontier count above")
        else:
            # A shaper that answered with no token counts is not a zero spend.
            print(f"\nshaper tokens: NOT REPORTED for {len(applied)} "
                  f"dispatch(es) -- the shaper returned no usage, so its spend "
                  f"is unknown rather than zero")
        if expansions:
            print(f"shaper expansion: {sum(expansions) / len(expansions):.2f}x "
                  f"the task's characters, so shaping ADDS prompt tokens to "
                  f"save completion tokens")
    # WHERE THE WORK WENT. `endpoint` was written into every result file and
    # read NOWHERE in this repo, so the summary named the model and never the
    # lane. Measured 2026-11-28 over the four saved A/B files on disk: 13 runs,
    # 12 to the hosted nous lane, 0 to any local GPU, and 1 whose file records
    # no endpoint at all. That is the DISTRIBUTE goal's defect -- one lane took
    # everything -- and it was invisible in the one place a person reads.
    #
    # The LOCAL count is the number that must be able to be non-zero. This stack
    # exists to keep bulk work on q38/the 3090, and a reader could not tell a
    # sweep that used them from one that never could.
    print(f"\nwrote {args.out}")
    # Print the endpoint the FILE records, not the live env var: they are the
    # same object today, and the file is the thing a reader re-opens.
    endpoint = results["endpoint"]
    print(f"endpoint: {endpoint}")
    # THREE outcomes, not two. The count used to be
    #   hosted = sum(1 for _ in results["runs"] if _is_hosted(ENDPOINT))
    # which folds "no recorded lane" into the hosted bucket -- _is_hosted
    # returns True for None, "" and any URL naming no host -- and had no local
    # sentence at all, because both branches required hosted != 0. So a sweep
    # to a card printed its endpoint and no count, and a sweep with
    # AB_ENDPOINT="" printed a full hosted sweep.
    lane = _lane_kind(endpoint)
    total = len(results["runs"])
    if lane == "unknown":
        # TWO causes, not one. An absent endpoint and a loopback port missing
        # from the placement tables are different problems with different fixes,
        # and the old sentence named only the first.
        unplaced = _unplaced_loopback_port(endpoint)
        if unplaced is not None:
            print(f"{total} run(s): LANE UNPLACED -- {endpoint!r} is a loopback "
                  f"tunnel on port {unplaced}, which is in neither "
                  f"LOCAL_LOOPBACK_PORTS nor HOSTED_LOOPBACK_PORTS, so this file "
                  f"cannot say whether a local GPU was used or a billed lane was "
                  f"charged")
        else:
            print(f"{total} run(s): LANE UNKNOWN -- no endpoint was recorded ({endpoint!r}),"
                  f" so this file cannot say whether any local GPU was used")
    elif lane == "local":
        print(f"all {total} run(s) went to a LOCAL lane; 0 hosted call(s) used")
    else:
        print(f"all {total} run(s) went to a HOSTED lane; 0 local GPU(s) used")
    return 0


if __name__ == "__main__":
    sys.exit(main())
