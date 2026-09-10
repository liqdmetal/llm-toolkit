# llm-toolkit

Free-portal LLM toolkit: local Ollama → Nous Portal → OpenRouter, all free.

Built and verified on a Windows RTX-3090 node running Hermes Agent. The
toolkit wires free LLM endpoints into scripts without burning paid credits —
local first (fast, private), aux fallbacks when local is down or the caller
opts into remote.

## What's in here

- `llm_client.py` — shared client. Local Ollama first; Nous Portal and
  OpenRouter as free aux. Router picks local, falls through to a free portal
  if local errored or the caller opts into remote. All three backends are
  free.
- `skills/hermes-agent/` — snapshot of the Hermes Agent skill tree as it was
  when this toolkit was assembled. Serves as a reference for how skills load,
  how the Hermes credential store is laid out, and how the agent surfaces
  (CLI, TUI, desktop, web, proxy) are configured.
- `reference/` — curated notes: auth-wiring pattern, the free-portal routing
  design, and the uncorked gateway smoke-test that proved the fallback end-to-end.
- `docs/` — supporting docs (auth wiring, design notes). No live secrets here.

## Quickstart

```bash
# 1. Make sure your Hermes credential store is in place.
#    This toolkit reads from ~/AppData/Local/hermes/auth.json (Windows) or
#    ~/.hermes/auth.json (Unix) by default. See docs/auth-wiring.md.

# 2. Drop llm_client.py on your PYTHONPATH (or the project dir).
python3 llm_client.py        # prints a status report
python3 -c "import llm_client; print(llm_client.status_report())"

# 3. Call it.
python3 - <<'PY'
import llm_client
r = llm_client.chat(
    messages=[{"role":"user","content":"Say OK"}],
    prefer_local=True,
    aux_portal="openrouter",
)
print(r)
PY
```

## Design

Priority ladder:

1. **Local Ollama** — Qwen 3.8 27B on the RTX 3090. Free, private, no network
   egress for inference. Primary.
2. **Nous Portal** — free models (`openai/gpt-5.6-luna`, `solar-pro-4-mini`,
   `deepseek-ai/deepseek-v4-flash`). Aux. Read from the Hermes credential
   store (OAuth token). Reachable at `https://inference-api.nousresearch.com/v1`.
3. **OpenRouter** — free models (`nvidia/nemotron-3-*-a12b:free`). Aux. API key
   from the Hermes credential store. Reachable at `https://openrouter.ai/api/v1`.

The router:

- Tries local first when `prefer_local=True` and Ollama is reachable.
- Falls through to the chosen aux portal (default OpenRouter) on local failure
  or when the caller opts into remote.
- If the chosen aux fails, tries the other free portal automatically.

Secrets are **never** committed. The client reads them at runtime from the
Hermes credential store or from env vars. See `docs/auth-wiring.md`.

## Verified behavior

- Local Ollama `llama3.2:3b` returns content.
- OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` returns content.
- Nous Portal `openai/gpt-5.6-luna` returns content.
- Router: local→OpenRouter fallback works; local→Nous fallback works.
- Gateway: when Ollama is down, a glass-tier key hits OpenRouter free model;
  owner tier gets 503 (no free-portal fallback for owner).

## License

MIT.
