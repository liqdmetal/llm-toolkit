# Comms-lane routing policy — llm-toolkit

Pipeline: user/ops prompt -> `shape.py` (defiant9b-abliterated on the 3090,
192.168.1.124:11434 — the same local box the Rosetta review finisher uses)
-> codebuff -> frontier model (claude-haiku-5.5 / gpt-6-luna-pro).

## Route rules (precedence top-down)

| Prompt class | Route |
|---|---|
| Reasoning-heavy / sensitive / creative analysis | LOCAL HERETIC MODELS (q38 jumbo, defiant abliterated, Qwen heretic on the 5090). Frontier not needed; refusals impossible if the model has no guardrails. |
| Frontier-only capability (Claude Code / codebuff work, long-context cross-check, model the locals can't do at all) | **Shaped** via `shape.py` first: task -> defiant9b-abliterated -> neutral decomposable prompt -> codebuff. Never send a raw emotional/adversarial ask. |
| Policy / terms / safety questions aimed AT the frontier model | Raw, unshaped. Do not shape policy questions — the answer must be the model's own. |
| Anything with byte-exact verifier (miner kernels, gates, golden tests) | Route to forge/q38 and let the GATE decide truth, not the model. Shaping optional. |

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
