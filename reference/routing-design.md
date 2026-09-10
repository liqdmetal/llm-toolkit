# Free-portal routing design

## Goal

Wire free LLM endpoints (Nous Portal, OpenRouter) into scripts so that:

1. Local Ollama is primary — free, private, no egress.
2. Free portals are aux — used when local is down or the caller opts into
   remote, without burning paid credits.
3. Credentials are read from the Hermes credential store at runtime — never
   committed.

## The client: llm_client.py

Single module. Imports cleanly from any script that puts the home dir on
`sys.path`. Exports:

- `local_chat(messages, model, base, max_tokens, temperature)` —
  calls local Ollama via `/api/generate` (the endpoint that works on this
  install; `/v1/chat/completions` times out for these GGUF models).
- `nous_chat(messages, model, max_tokens, temperature)` — calls Nous Portal
  `/chat/completions` with the OAuth bearer token.
- `openrouter_chat(messages, model, max_tokens, temperature)` — calls
  OpenRouter `/chat/completions` with the API key.
- `chat(...)` — router. Tries local first (if reachable), then aux portal,
  then the other aux portal if the first failed.
- `aux_chat(messages, portal, model, ...)` — direct call to one portal.
- `status_report()` — diagnostics: which backends are reachable, whether
  credentials were found.

## Credential model

- Read from `~/AppData/Local/hermes/auth.json` (Windows) or `~/.hermes/auth.json`
  (Unix) at import time.
- Nous: OAuth bearer token under `providers.nous.access_token`.
- OpenRouter: API key under `credential_pool.openrouter[0].access_token`.
- Env vars override (`OPENROUTER_API_KEY`, `NOUS_INFERENCE_URL`, etc.).

## Why /api/generate, not /v1/chat/completions

The local Ollama instance on this node serves `/api/generate` reliably for the
GGUF models in use (Qwen 3.8 27B, llama3.2:3b). The `/v1/chat/completions`
OpenAI-compat layer times out for these models on this install. `llm_client.py`
uses `/api/generate` and converts chat messages to a prompt string.

Known limitation: when the 27B model is still loading into VRAM,
`/api/generate` returns `done_reason: "load"` with an empty response. The client
treats that as a failure so the router falls through to aux. A caller that
wants to wait for the model to load can retry locally.

## Router behavior

```
chat(messages, prefer_local=True, aux_portal="openrouter")
  → if local reachable and prefer_local:
      call local_chat
      → if ok: return
      → if not ok: fall through to aux
  → call aux_chat(port=aux_portal)
      → if ok: return
      → if not ok: try the other portal
      → if both fail: return error
```

## Integration points

### Scripts that use llm_client.py

- `cpa-ais-starter/engine/ai.py` — patched `_llm()` to route through the
  client. `"off"` provider routes local→free aux. `"ollama"` falls back to
  free portal when local fails. `"openai"` also routes to free portals if
  `LS_AI_BASE` points at them.
- `fullbore/warwagon/warwagon.py` — patched `llm_analyze()` to call
  `chat()` with free-portal aux. Remote consent gated by `--allow-remote-llm`.
- `uncorked/gateway.py` — patched `_backend_dispatch()` to fall back to
  `aux_chat()` (OpenRouter `nemotron-3-super-120b-a12b:free` or Nous
  `gpt-5.6-luna`) when Ollama is down. Non-owner tiers get the fallback;
  owner tier keeps its own paid fallback.

### The uncorked gateway smoke test

The gateway is a Tailnet-exposed LLM proxy with tiered access (glass/bottle/case/
owner). The smoke test verifies:

1. Healthz shows Ollama status.
2. A glass-tier key can chat (local Ollama).
3. A glass-tier key can stream (local Ollama).
4. Kill Ollama → glass-tier key falls back to a free portal (OpenRouter by
   default) and gets content.
5. Owner tier gets 503 when Ollama is down (no free-portal fallback).

The gateway code that implements this is `gateway.py` (in the `uncorked/`
project), specifically the `_backend_dispatch()` method.

## Limitations

- The client reads credentials once at import time. If a key rotates, restart
  the process.
- The client does not cache or retry across process boundaries.
- The local `/api/generate` path does not support multi-turn conversation with
  assistant history as richly as the OpenAI-compat path — it uses the last user
  message (plus system, if present). For multi-turn, route to a portal.
- Streaming from aux portals through the gateway works (non-stream verified
  live; stream verified live against local Ollama; the free-portal stream path
  was patched to fix a header-leak bug).
