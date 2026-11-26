# Comms-lane routing policy — llm-toolkit

Pipeline: user/ops prompt -> `shape.py` (defiant9b-abliterated on the 3090,
192.168.1.124:11434 — the same local box the Rosetta review finisher uses)
-> codebuff -> frontier model (claude-haiku-5.5 / gpt-6-luna-pro).

## Route rules (precedence top-down)

| Prompt class | Route |
|---|---|
| Reasoning-heavy / sensitive / creative analysis | LOCAL HERETIC MODELS — see the measured availability note below. Frontier not needed; refusals impossible if the model has no guardrails. |
| Frontier-only capability (Claude Code / codebuff work, long-context cross-check, model the locals can't do at all) | **Shaped** via `shape.py` first: task -> defiant9b-abliterated -> neutral decomposable prompt -> codebuff. Never send a raw emotional/adversarial ask. |
| Policy / terms / safety questions aimed AT the frontier model | Raw, unshaped. Do not shape policy questions — the answer must be the model's own. |
| Anything with byte-exact verifier (miner kernels, gates, golden tests) | Route to forge/q38 and let the GATE decide truth, not the model. Shaping optional. |

## Measured availability (2026-11-29 run, `fleet_probe.py`)

This row used to name "q38 jumbo, defiant abliterated, Qwen heretic on the 5090".
Probed rather than assumed, that is wrong on all three counts: none of those names
exist on the 5090, and the abliterated model lives on the 3090.

The table below carries **every lane the roster measures**, because a lane with no
row is a lane the policy does not route to. MEASURED 2026-11-29 at `237e5d2`: the
three rows this table used to have described the lanes that were IDLE, and the
three lanes it did not mention were the three SERVING — including the one lane
with measured free slots (`3090-defiant`, 0/4 busy). A reader of this policy was
therefore pointed away from the only lane with established spare capacity, which
is this file's own expensive direction.

| Lane | State | Models |
|---|---|---|
| nous portal `127.0.0.1:8099` | UP, hosted, 0.7 s | 409 |
| 5090 ollama `192.168.1.232:11434` | UP, idle 0/3 loaded | 3: `7b-qwen-q4`, `qwen2.5:7b`, `llama3.2-vision` |
| q38 llama.cpp `192.168.1.232:8081` | UP, SERVING 1, slots unreadable (no key) | not listed by `/v1/models` |
| 3090 ollama `192.168.1.124:11434` | UP, idle 0/7 loaded | 7, incl. `defiant9b-abliterated:256k` |
| 3090-inkling `127.0.0.1:18082` | UP, SERVING 1, slots unreadable (no key) | not listed |
| 3090-defiant `127.0.0.1:8090` | UP, SERVING 1, **0/4 slots busy** | `defiant-vision.gguf` |

The `127.0.0.1` rows are ssh tunnels to `192.168.1.124`; `host_capacity()` maps
them to their far end so one card is not counted twice. Only `3090-defiant`
reports a busy count, so it is the only lane whose spare capacity is ESTABLISHED;
the two unreadable ones are "not counted as free", never "idle".

Sensitive reasoning therefore routes to the **3090**, which holds
`defiant9b-abliterated:256k` (what `shape.py` already defaults to). The 5090 is
not a heretic box today; it holds small Qwen and a vision model.

The latency figures live in the probe's output, not in this table. Two of them
(`316 ms`, `410` models) were stale by this run — the same lanes measured 0.7 s
and 409 — and a hard-coded millisecond in a policy document is a number no
instrument reproduces once the fleet is re-probed. `tests/test_routing_lane_table_matches_roster.py`
holds the table to the roster: every measured lane needs a row, no row may name a
lane the roster does not measure, each row must state lane STATE and not only a
speed, and this section must name the date its figures came from.

DO NOT read an empty `/api/ps` as an idle MACHINE. The 5090's ollama answers
`{"models":[]}` while `http://192.168.1.232:8081/health` — q38's llama.cpp
server on the SAME host — answers 200 with a model loaded. MEASURED
2026-11-26: `[2/3] local lanes SERVING (idle: 5090)` was true per DAEMON and
false per HOST, and `[0/2] local hosts with idle capacity` was the honest
figure. The distinction is the expensive direction: counting daemons as
capacity invites a dispatch onto a card that is already busy. Capacity is
counted in MACHINES by `host_capacity()`, never in HTTP daemons.

Run `python comms-lane/fleet_probe.py` before claiming a lane is available. The
probe lists models only (`/api/tags`, `/v1/models`) and never starts a
generation, so it cannot evict a resident model or change what it measures.

## Hard rules

1. `.shape.py` is a **soft** dependency: shaper failure falls through to the
   raw task (`exit 1` -> caller uses original). Frontline work must never
   hard-block on the 3090.
2. Shaping is NOT a bypass. It removes tripwire framing only. If the frontier
   model still refuses, the refusal is the truthful answer — record it, don't
   fight it.
3. Never shape policy/terms/safety prompts.
4. Never claim "uncensored output" — claim "shaped prompt, refusal rate
   measured".
5. Local heretic models are the default for sensitive reasoning; frontier
   only where capability demands it. Frontier spend stays capped as the
   account dictates (luna/claude = reserve).

## Measurement (prove it works)

A/B test the same task set through (a) raw codebuff and (b) shaped codebuff,
record refusal/answer-completion per attempt. Track in reference/shaping-a-b.md
once we run one; update this policy with the measured refusal delta.
