# Auth wiring — how llm_client.py gets its credentials

This file documents **how** credentials are wired. It does **not** contain
live credentials. The actual credentials live in the Hermes credential store
and are never committed to this repo.

## Credential sources (in priority order)

1. **Environment variables** — highest priority, override everything.
   - `NOUS_INFERENCE_URL` — override the Nous inference base URL
   - `OPENROUTER_API_KEY` — override the OpenRouter API key
   - `NOUS_FREE_MODEL` — override the default Nous free model
   - `OPENROUTER_FREE_MODEL` — override the default OpenRouter free model

2. **Hermes credential store** — `~/AppData/Local/hermes/auth.json` on Windows,
   `~/.hermes/auth.json` on Unix. This is where Hermes keeps live OAuth tokens
   and API keys. The path is configurable via `HERMES_HOME`.

## What llm_client.py reads

### Nous Portal

From `auth.json`, under `providers.nous`:

- `access_token` — the OAuth bearer token used for `Authorization: Bearer ...`
- `inference_base_url` — override for the inference base URL (default:
  `https://inference-api.nousresearch.com/v1`)
- `expires_at` — token expiry timestamp (informational)

The token is refreshed by Hermes itself; llm_client.py just reads the current
value. When the token expires, the caller (Hermes) refreshes it and llm_client
picks up the new value on the next import.

### OpenRouter

From `auth.json`, under `credential_pool.openrouter` (a list; the first entry
is used):

- `access_token` — the `sk-or-v1-...` API key

Or from the env var `OPENROUTER_API_KEY` if set.

## File layout (Windows, default)

```
C:/Users/<user>/AppData/Local/hermes/
  auth.json          ← credentials (NEVER committed)
  config.yaml        ← provider/model config (OK to commit if scrubbed)
  skills/            ← installed skills
  plugins/           ← installed plugins
```

## How scripts find llm_client.py

`llm_client.py` lives in the user's home directory (e.g.
`C:/Users/<user>/llm_client.py`). Scripts that use it either:

- Add the home directory to `sys.path` before importing:
  ```python
  import sys, os
  _HD = os.path.expanduser("~")
  if _HD not in sys.path:
      sys.path.insert(0, _HD)
  from llm_client import chat, aux_chat, local_chat, ollama_available
  ```

- Or run with the home directory on `PYTHONPATH`:
  ```bash
  set PYTHONPATH=%USERPROFILE%;%PYTHONPATH%
  python script.py
  ```

## Scrubbing config.yaml before committing

`config.yaml` may reference model names and provider config. It does **not**
contain secrets (those are in `auth.json`). If you commit a scrubbed
`config.yaml`, make sure any model-specific tuning parameters are intended for
sharing. When in doubt, omit it.

## Rotating keys

- **OpenRouter**: regenerate the key in the OpenRouter dashboard, update
  `auth.json` (or set `OPENROUTER_API_KEY`), and restart any processes that
  import `llm_client.py` (the module reads the key once at import time).

- **Nous Portal**: the OAuth token is refreshed automatically by Hermes. If you
  need to force a refresh, restart Hermes — it re-authorizes on startup.

## Testing the wiring

```bash
python3 llm_client.py
```

This prints a status report showing whether each backend is reachable and
whether credentials were found. If a backend shows as unavailable, check:

1. Is the credential store at the expected path?
2. Does `auth.json` contain the expected keys?
3. Are the env vars overriding something you didn't intend?
