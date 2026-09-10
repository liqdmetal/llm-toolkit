#!/usr/bin/env python3
"""llm_client.py - shared LLM client: local Ollama first, free portals as aux.

Priority ladder:
  1. Local Ollama  (qwen3.8-uncensored:27b on RTX 3090) - primary, free, private
  2. Nous Portal   (free models: openai/gpt-5.6-luna, solar-pro-4-mini,
                     deepseek-v4-flash) - aux, free for subscribers
  3. OpenRouter    (free models: nvidia/nemotron-*) - aux, free models

All three are free. The router picks local first (fast, private), then falls
back to a free portal if local is down or the caller opts into remote.

Secrets: read from env vars or ~/.hermes/auth.json (Hermes credential store).
Env vars (override auth.json):
  NOUS_INFERENCE_URL  - override inference base URL
  OPENROUTER_API_KEY  - override OpenRouter key
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

HERMES_HOME = Path(os.environ.get(
    "HERMES_HOME",
    Path.home() / "AppData" / "Local" / "hermes"))
AUTH_PATH = HERMES_HOME / "auth.json"


def _read_auth():
    try:
        with open(AUTH_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_AUTH = _read_auth()


NOUS_INFERENCE_BASE = os.environ.get(
    "NOUS_INFERENCE_URL",
    _AUTH.get("providers", {}).get("nous", {}).get(
        "inference_base_url",
        "https://inference-api.nousresearch.com/v1"),
).rstrip("/")

_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY") or (
    _AUTH.get("credential_pool", {}).get("openrouter", [{}])[0].get(
        "access_token", "")
)
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def nous_free_model():
    return os.environ.get("NOUS_FREE_MODEL", "openai/gpt-5.6-luna")


def openrouter_free_model():
    return os.environ.get("OPENROUTER_FREE_MODEL",
                           "nvidia/nemotron-3-super-120b-a12b:free")


def ollama_available(base="http://localhost:11434"):
    try:
        req = urllib.request.Request(
            f"{base}/api/tags",
            headers={"User-Agent": "llm_client/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def local_chat(messages, model="qwen3.8-uncensored:27b",
               base="http://localhost:11434", max_tokens=4096,
               temperature=0.7, **kwargs):
    """Call local Ollama via /api/generate (the endpoint that works on this install).

    /v1/chat/completions times out on this Ollama instance for these GGUF models;
    /api/generate returns valid JSON. Converts chat messages to a prompt string.
    """
    base = base.rstrip("/")
    # Build a prompt from messages. Ollama /api/generate takes a single prompt
    # string. Use the last user message; if there's a system message, prepend it.
    system = ""
    user_content = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "system":
            system = content
        elif role in ("user", "assistant"):
            user_content = content
    # If no user message found, join all non-system messages as prompt
    if not user_content:
        user_content = " ".join(
            m.get("content", "") or "" for m in messages
            if m.get("role") != "system")
    prompt = f"{system}\n\n{user_content}" if system else user_content
    url = f"{base}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "User-Agent": "llm_client/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
        content = resp.get("response", "")
        # Ollama /api/generate can return empty response when model is still
        # loading (done_reason='load') - treat as failed so the router falls
        # through to aux.
        if not content and resp.get("done_reason") == "load":
            return {"ok": False, "error": "model still loading (VRAM)",
                    "provider": "local"}
        return {
            "ok": True,
            "content": content,
            "provider": "local",
            "model": model,
            "tokens": {"prompt": len(prompt) // 4,
                       "completion": len(content) // 4},
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "provider": "local"}


def _nous_auth_header():
    tok = _AUTH.get("providers", {}).get("nous", {}).get(
        "access_token", "")
    if not tok:
        return {}
    return {"Authorization": f"Bearer {tok}"}


def nous_chat(messages, model=None, max_tokens=4096,
              temperature=0.7, **kwargs):
    if model is None:
        model = nous_free_model()
    url = f"{NOUS_INFERENCE_BASE}/chat/completions"
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json",
               "User-Agent": "llm_client/1.0"}
    headers.update(_nous_auth_header())
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
        content = (resp.get("choices") or [{}])[0].get(
            "message", {}).get("content", "")
        return {"ok": True, "content": content, "provider": "nous",
                "model": model, "tokens": resp.get("usage", {})}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        return {"ok": False, "error": f"HTTP {e.code}: {body_text[:300]}",
                "provider": "nous"}
    except Exception as e:
        return {"ok": False, "error": str(e), "provider": "nous"}


def openrouter_chat(messages, model=None, max_tokens=4096,
                    temperature=0.7, **kwargs):
    if model is None:
        model = openrouter_free_model()
    url = f"{OPENROUTER_BASE}/chat/completions"
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json",
               "User-Agent": "llm_client/1.0",
               "Authorization": f"Bearer {_OPENROUTER_KEY}"}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
        content = (resp.get("choices") or [{}])[0].get(
            "message", {}).get("content", "")
        return {"ok": True, "content": content, "provider": "openrouter",
                "model": model, "tokens": resp.get("usage", {})}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        return {"ok": False, "error": f"HTTP {e.code}: {body_text[:300]}",
                "provider": "openrouter"}
    except Exception as e:
        return {"ok": False, "error": str(e), "provider": "openrouter"}


def chat(messages, prefer_local=True, max_tokens=4096, temperature=0.7,
         aux_portal="openrouter", aux_model=None, model=None,
         local_model=None, local_base="http://localhost:11434",
         local_first=True):
    local_m = local_model or model or "qwen3.8-uncensored:27b"
    aux_m = aux_model or model

    if local_first and prefer_local:
        if ollama_available(local_base):
            r = local_chat(messages, model=local_m, base=local_base,
                           max_tokens=max_tokens, temperature=temperature)
            if r.get("ok"):
                return r
            # local errored - fall through to aux

    if aux_portal and aux_portal != "none":
        if aux_portal == "nous":
            r = nous_chat(messages, model=aux_m, max_tokens=max_tokens,
                          temperature=temperature)
        elif aux_portal == "openrouter":
            r = openrouter_chat(messages, model=aux_m,
                                max_tokens=max_tokens,
                                temperature=temperature)
        else:
            return {"ok": False, "error": f"unknown aux_portal: {aux_portal}"}
        if r.get("ok"):
            return r
        # aux failed - try the other free portal
        other = "openrouter" if aux_portal == "nous" else "nous"
        r2 = (openrouter_chat if other == "openrouter" else nous_chat)(
            messages, model=aux_m, max_tokens=max_tokens,
            temperature=temperature)
        if r2.get("ok"):
            return {"ok": True, "content": r2["content"],
                    "provider": other + "(fallback)", "model": aux_m,
                    "tokens": r2.get("tokens", {})}
        return {"ok": False, "error": r.get("error") or r2.get(
            "error", "both portals failed")}

    return {"ok": False, "error": "no available LLM endpoint"}


def aux_chat(messages, portal="openrouter", model=None,
             max_tokens=4096, temperature=0.7, **kwargs):
    if portal == "nous":
        return nous_chat(messages, model=model, max_tokens=max_tokens,
                         temperature=temperature, **kwargs)
    if portal == "openrouter":
        return openrouter_chat(messages, model=model,
                               max_tokens=max_tokens,
                               temperature=temperature, **kwargs)
    return {"ok": False, "error": f"unknown portal: {portal}"}


def status_report():
    local_ok = ollama_available()
    Nous_ok = bool(_AUTH.get("providers", {}).get("nous", {}).get(
        "access_token"))
    OR_ok = bool(_OPENROUTER_KEY)
    return {
        "local_ollama": local_ok,
        "nous_portal": Nous_ok,
        "openrouter": OR_ok,
        "nous_token_expires": _AUTH.get("providers", {}).get(
            "nous", {}).get("expires_at"),
        "openrouter_key_prefix": (_OPENROUTER_KEY[:8] + "...") if _OPENROUTER_KEY else None,
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(status_report(), indent=2))
