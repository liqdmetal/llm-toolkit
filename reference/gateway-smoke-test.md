# Gateway smoke test — free-portal fallback proof

This directory reproduces the smoke test that proved the free-portal fallback
works end-to-end. The gateway under test is `gateway.py` (an uncorked-tailnet
LLM proxy with tiered access: glass/bottle/case/owner).

## What it proves

1. The gateway starts and serves on a loopback port.
2. A glass-tier key can chat against local Ollama (non-stream + stream).
3. When Ollama is killed, the same glass-tier key falls back to a free portal
   (OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` by default) and gets
   content.
4. An owner-tier key gets **503** when Ollama is down — no free-portal fallback
   for owner (they bring their own paid endpoint via `FALLBACK_BASE`/`FALLBACK_KEY`).

## Usage

```bash
# 1. Start the gateway
python gateway.py server --port 11597 &

# 2. Wait for it to be ready
sleep 5
curl -s http://localhost:11597/healthz

# 3. Register a glass-tier key
python gateway.py issue "glass user" glass

# 4. Register an owner-tier key
python gateway.py issue "owner" owner

# 5. Chat as glass (local Ollama)
curl -s -X POST http://localhost:11597/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <glass-key>" \
  -d '{"model":"qwen3.8-uncensored:27b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5,"temperature":0}'

# 6. Kill Ollama
#    On Windows: taskkill /F /PID <ollama-pid>
#    Find the PID: netstat -ano | findstr 11434

# 7. Chat as glass again — should fall back to OpenRouter free model
curl -s -X POST http://localhost:11597/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <glass-key>" \
  -d '{"model":"qwen3.8-uncensored:27b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5,"temperature":0}'

# 8. Chat as owner — should 503
curl -s -X POST http://localhost:11597/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <owner-key>" \
  -d '{"model":"qwen3.8-uncensored:27b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5,"temperature":0}'

# 9. Teardown
kill %1
```

## Expected results

- Step 2: `{"ok":true,"ollama":true,"fallback_configured":false}` (or
  `"ollama":false` if Ollama is already down).
- Step 5: A chat completion with `"content":"OK"` (or whatever you asked for).
- Step 7: A chat completion with model `"nvidia/nemotron-3-super-120b-a12b:free"`
  and content matching your prompt. This is the free-portal fallback.
- Step 8: `{"error":{"message":"cellar down: ...","type":"server_error",...}}`
  — 503, because the owner tier has no budget and no free-portal fallback.

## Configuring the fallback portal

Set these env vars before starting the gateway:

```bash
# Which free portal to use for non-owner fallback (default: openrouter)
export UNCORKED_AUX_PORTAL=nous   # or openrouter

# Explicit model override (default: portal's best free model)
export UNCORKED_AUX_MODEL=openai/gpt-5.6-luna
```

When `UNCORKED_AUX_PORTAL=nous`, the fallback model is `openai/gpt-5.6-luna`.
When `UNCORKED_AUX_PORTAL=openrouter` (default), it's
`nvidia/nemotron-3-super-120b-a12b:free`.

## Credentials

The gateway reads the same Hermes credential store as `llm_client.py`:

- `~/AppData/Local/hermes/auth.json` (Windows) or `~/.hermes/auth.json` (Unix).
- Nous: OAuth bearer token under `providers.nous.access_token`.
- OpenRouter: API key under `credential_pool.openrouter[0].access_token`.

See `docs/auth-wiring.md` in the parent directory.

## Notes

- The gateway imports `llm_client.py` from the home directory. Make sure
  `llm_client.py` is on the Python path (home dir on `sys.path`).
- The gateway uses `ThreadingHTTPServer` on loopback only. It is not exposed
  to the internet directly — transport is Tailscale.
- The gateway uses SQLite (WAL mode) for the key/usage store. The DB path is
  `uncorked.db` in the same directory as `gateway.py`.
- Budget resets monthly (UTC). Glass tier: 750,000 tokens/month. Bottle:
  2,000,000. Case: 5,000,000. Owner: unlimited (no budget, no free-portal
  fallback).
