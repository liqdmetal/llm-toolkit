#!/usr/bin/env python3
# UNCORKED gateway — private uncensored LLM over Tailnet.
# Stdlib only. Loopback bind; Tailscale serve is the transport.
import argparse, json, os, sqlite3, threading, time
import urllib.request, urllib.error
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "uncorked.db"
OLLAMA = os.environ.get("UNCORKED_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_BASE = os.environ.get("UNCORKED_FALLBACK_BASE_URL", "").rstrip("/")
FALLBACK_KEY = os.environ.get("UNCORKED_FALLBACK_KEY", "")
FALLBACK_MODEL = os.environ.get("UNCORKED_FALLBACK_MODEL", "deepseek/deepseek-chat")

# Ensure llm_client (in home dir) is importable regardless of CWD.
_HD = os.path.expanduser("~")
if _HD not in sys.path:
    sys.path.insert(0, _HD)

# Free-portal opts (configured via env, used when Ollama is down).
#   UNCORKED_AUX_PORTAL = nous | openrouter   (default: openrouter)
#   UNCORKED_AUX_MODEL  = explicit model name (default: portal's best free)
# These are read inside _backend_dispatch / call_fallback_chat via llm_client.
#
try:
    from llm_client import aux_chat, chat, ollama_available as _local_ollama_up
except ImportError:
    _local_ollama_up = None   # type: ignore[assignment]
    aux_chat = None           # type: ignore[assignment]
    chat = None               # type: ignore[assignment]

UNCORKED_AUX_PORTAL = os.environ.get("UNCORKED_AUX_PORTAL", "openrouter")
UNCORKED_AUX_MODEL = os.environ.get("UNCORKED_AUX_MODEL", "")

HOST = os.environ.get("UNCORKED_HOST", "127.0.0.1")
PORT = int(os.environ.get("UNCORKED_PORT", "11500"))
DEFAULT_MODEL = "qwen3.8-uncensored:27b"
ALLOWED_MODELS = {"qwen3.8-uncensored:27b", "qwen3.8:27b", "llama3.2:3b"}
SUB_INFLIGHT = 2      # global in-flight sub streams
QUEUE_MAX = 4         # waiting room
UPSTREAM_TIMEOUT = 900
TIER_BUDGET = {"glass": 750_000, "bottle": 2_000_000, "case": 5_000_000, "owner": None}

_local = threading.local()
DB_LOCK = threading.Lock()
inflight = threading.BoundedSemaphore(SUB_INFLIGHT)
owner_slot = threading.Semaphore(1)
pending = threading.BoundedSemaphore(QUEUE_MAX)

def db():
    c = getattr(_local, "c", None)
    if c is None:
        c = sqlite3.connect(str(DB_PATH))
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        _local.c = c
    return c

def init_db():
    with DB_LOCK:
        c = db()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS keys(
          key TEXT PRIMARY KEY, name TEXT, tier TEXT,
          active INTEGER DEFAULT 1, created TEXT);
        CREATE TABLE IF NOT EXISTS ledger(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, ts TEXT,
          model TEXT, prompt_tokens INTEGER, comp_tokens INTEGER);
        CREATE TABLE IF NOT EXISTS credits(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, ts TEXT,
          tokens INTEGER, note TEXT);
        CREATE TABLE IF NOT EXISTS packs(
          id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, ts TEXT,
          plan TEXT, usd REAL, method TEXT);
        """)
        c.commit()

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def month_now():
    return time.strftime("%Y-%m", time.gmtime())

def usage_month(key):
    c = db()
    used = c.execute(
        "SELECT COALESCE(SUM(comp_tokens),0) FROM ledger WHERE key=? AND substr(ts,1,7)=?",
        (key, month_now())).fetchone()[0]
    cred = c.execute(
        "SELECT COALESCE(SUM(tokens),0) FROM credits WHERE key=? AND substr(ts,1,7)=?",
        (key, month_now())).fetchone()[0]
    return max(0, used - cred), used, cred

def acquire_slot(key, is_owner):
    if is_owner:
        if not owner_slot.acquire(timeout=0):
            return False
    else:
        if not pending.acquire(timeout=5):
            return False
        if not inflight.acquire(timeout=30):
            pending.release()
            return False
    return True

def release_slot(key, is_owner):
    if is_owner:
        owner_slot.release()
    else:
        inflight.release()
        pending.release()

def record_usage(key, model, prompt_tokens, comp_tokens):
    with DB_LOCK:
        c = db()
        c.execute("INSERT INTO ledger(key,ts,model,prompt_tokens,comp_tokens) VALUES(?,?,?,?,?)",
                  (key, now_iso(), model, prompt_tokens, comp_tokens))
        c.commit()

def _upstream(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
    except urllib.error.URLError as e:
        raise ConnectionError(f"{url}: {e.reason}") from e
    except (TimeoutError, OSError) as e:
        raise ConnectionError(f"{url}: {e}") from e

def call_ollama_chat(model, messages, stream, options):
    payload = {"model": model, "messages": messages, "stream": stream, "keep_alive": "24h"}
    if options:
        opts = {}
        for k in ("temperature", "top_p", "top_k", "repeat_penalty"):
            if k in options: opts[k] = options[k]
        if "max_tokens" in options: opts["num_predict"] = options["max_tokens"]
        if opts: payload["options"] = opts
    return _upstream(f"{OLLAMA}/api/chat", payload, {"Content-Type": "application/json"})

def call_ollama_raw(path, payload):
    return _upstream(f"{OLLAMA}{path}", payload, {"Content-Type": "application/json"})

def call_fallback_chat(model, messages, stream, options):
    body = {
        "model": FALLBACK_MODEL, "messages": messages, "stream": stream,
    }
    for k in ("temperature", "top_p", "max_tokens"):
        if k in (options or {}): body[k] = options[k]
    if stream: body["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {FALLBACK_KEY}"}
    return _upstream(f"{FALLBACK_BASE}/chat/completions", body, headers)

def est_tokens(text):
    return max(1, len(text or "") // 4)

# ---------- HTTP ----------

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # zero-retention: no access log noise

    def _send(self, code, obj, raw=None, ctype="application/json", extra=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_err(self, code, err, etype="invalid_request_error"):
        self._send(code, {"error": {"message": err, "type": etype, "code": None}})

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _auth(self):
        h = self.headers.get("Authorization", "")
        if not h.startswith("Bearer "):
            return None, None
        key = h[7:].strip()
        row = db().execute("SELECT * FROM keys WHERE key=?", (key,)).fetchone()
        if row and row["active"]:
            return key, row
        return None, None

    def do_OPTIONS(self):
        self._send(204, None, raw=b"")

    def do_GET(self):
        if self.path == "/healthz":
            ollama_ok = False
            try:
                with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2) as r:
                    ollama_ok = r.status == 200
            except Exception:
                pass
            return self._send(200, {"ok": True, "ollama": ollama_ok,
                                   "fallback_configured": bool(FALLBACK_BASE and FALLBACK_KEY)})
        if self.path in ("/v1/models", "/models"):
            keyinfo = self._auth()
            if not keyinfo[0]:
                return self._json_err(401, "missing or unknown API key")
            return self._send(200, {
                "object": "list",
                "data": [{"id": m, "object": "model", "created": int(time.time()), "owned_by": "uncorked"}
                         for m in sorted(ALLOWED_MODELS)]})
        return self._json_err(404, "not found")

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json_err(404, "not found")
        keyinfo = self._auth()
        if not keyinfo[0]:
            return self._json_err(401, "missing or unknown API key")
        key, row = keyinfo
        tier = row["tier"]
        budget = TIER_BUDGET.get(tier)
        if budget is not None:
            used, used_raw, credits = usage_month(key)
            if used >= budget:
                return self._json_err(429, "Monthly barrel is empty (budget reached). Resets next month UTC.",
                                      "insufficient_quota")
        try:
            body = self._read_body()
        except Exception:
            return self._json_err(400, "invalid JSON body")
        messages = body.get("messages")
        if not isinstance(messages, list):
            return self._json_err(400, "`messages` must be a list")
        model = body.get("model", DEFAULT_MODEL)
        if model not in ALLOWED_MODELS:
            return self._json_err(400, f"model not in cellar: {model}")
        stream = bool(body.get("stream"))
        options = body.get("options") or {}

        is_owner = (tier == "owner")
        if not acquire_slot(key, is_owner):
            return self._json_err(429, "The cellar is full — try again in a moment.", "rate_limit")
        try:
            self._chat(key, model, messages, stream, options, budget)
        except ConnectionError as e:
            self._backend_dispatch(key, model, messages, stream, options, budget, first_err=str(e))
        except Exception as e:
            self._json_err(502, f"upstream error: {e}", "server_error")
        finally:
            release_slot(key, is_owner)

    def _backend_dispatch(self, key, model, messages, stream, options, budget, first_err=None):
        """Try Ollama; on connection failure fall back to free portals
        (Nous, OpenRouter) so subs never block on model availability.

        Tiers with a budget (glass/bottle/case) get free-portal fallback
        when Ollama is down. The owner tier has no budget and gets no
        free-portal fallback (they pay for their own cloud endpoint if
        they set FALLBACK_BASE/KEY).
        """
        try:
            self._chat(key, model, messages, stream, options, budget)
            return
        except Exception as e2:
            first_err = first_err or str(e2)
        # 1) Configured fallback (paid/enterprise endpoint if set) — available
        #    to all tiers including owner.
        if FALLBACK_BASE and FALLBACK_KEY:
            try:
                self._chat_fallback(key, model, messages, stream, options)
                return
            except Exception as e3:
                first_err = f"{first_err}; failover: {e3}"
        # 2) Free portals — only for non-owner tiers (those with a budget),
        #    only if llm_client available. Owner tier has no budget and already
        #    got the configured fallback above.
        if budget is not None and aux_chat is not None:
            try:
                model_name = model if model in ALLOWED_MODELS else DEFAULT_MODEL
                portal = UNCORKED_AUX_PORTAL
                pm = UNCORKED_AUX_MODEL or (
                    "openai/gpt-5.6-luna" if portal == "nous" else "nvidia/nemotron-3-super-120b-a12b:free"
                )
                r = aux_chat(
                    messages=messages,
                    portal=portal,
                    model=pm,
                    max_tokens=options.get("max_tokens", 4096),
                    temperature=options.get("temperature", 0.7),
                )
                if r.get("ok"):
                    # Stream or non-stream back through the gateway.
                    if stream:
                        self._send_start("text/event-stream")
                        chunk_id = f"chatcmpl-{int(time.time()*1000)}"
                        self._send_chunk(chunk_id, r["content"], True, False, 0, 0)
                        epilogue = {"id": chunk_id, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": pm,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                        self.wfile.write(b"data: " + json.dumps(epilogue).encode() + b"\n\n")
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        self._send(200, {
                            "id": f"chatcmpl-{int(time.time()*1000)}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": pm,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": r["content"]},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        })
                    return
                first_err = f"{first_err}; free-portal {portal}: {r.get('error', 'failed')}"
            except Exception as e3:
                first_err = f"{first_err}; free-portal {portal} error: {e3}"
        self._json_err(503, f"cellar down: {first_err}", "server_error")

    def _open_upstream(self, key, model, messages, stream, options):
        if stream:
            resp = call_ollama_chat(model, messages, True, options)
        else:
            resp = call_ollama_chat(model, messages, False, options)
        return resp

    def _chat(self, key, model, messages, stream, options, budget):
        if stream:
            self._send_start("text/event-stream")
            chunk_id = f"chatcmpl-{int(time.time()*1000)}"
            finish = time.time()
            content_parts = []
            prompt_tokens = comp_tokens = 0
            with self._open_upstream(key, model, messages, True, options) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("error"):
                        raise ConnectionError(obj["error"])
                    piece = (obj.get("message") or {}).get("content") or ""
                    if piece:
                        content_parts.append(piece)
                        ch = {"id": chunk_id, "object": "chat.completion.chunk",
                              "created": int(finish), "model": model,
                              "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                        self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
                    if obj.get("done"):
                        prompt_tokens = obj.get("prompt_eval_count") or 0
                        comp_tokens = obj.get("eval_count") or len("".join(content_parts)) // 4
            if not comp_tokens:
                comp_tokens = est_tokens("".join(content_parts))
            if not prompt_tokens:
                prompt_tokens = est_tokens(json.dumps(messages))
            final = {"id": chunk_id, "object": "chat.completion.chunk", "created": int(finish), "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
                               "total_tokens": prompt_tokens + comp_tokens}}
            self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            record_usage(key, model, prompt_tokens, comp_tokens)
        else:
            with self._open_upstream(key, model, messages, False, options) as r:
                obj = json.loads(r.read().decode())
            if obj.get("error"):
                raise ConnectionError(obj["error"])
            content = (obj.get("message") or {}).get("content") or ""
            ts = int(time.time())
            out = {"id": f"chatcmpl-{ts}", "object": "chat.completion", "created": ts, "model": model,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": obj.get("prompt_eval_count") or est_tokens(json.dumps(messages)),
                             "completion_tokens": obj.get("eval_count") or est_tokens(content),
                             "total_tokens": (obj.get("prompt_eval_count") or 0) + (obj.get("eval_count") or 0)}}
            if out["usage"]["total_tokens"] == 0:
                out["usage"] = {"prompt_tokens": est_tokens(json.dumps(messages)),
                                "completion_tokens": est_tokens(content),
                                "total_tokens": out["usage"]["prompt_tokens"] + out["usage"]["completion_tokens"]}
            self._send(200, out)
            record_usage(key, model, out["usage"]["prompt_tokens"], out["usage"]["completion_tokens"])

    def _send_start(self, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _chat_fallback(self, key, model, messages, stream, options):
        if stream:
            self._send_start("text/event-stream")
            prompt_tokens = comp_tokens = 0
            content_parts = []
            with call_fallback_chat(model, messages, True, options) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        o = json.loads(data)
                    except Exception:
                        continue
                    if o.get("usage"):
                        prompt_tokens = o["usage"].get("prompt_tokens", 0)
                        comp_tokens = o["usage"].get("completion_tokens", 0)
                        continue
                    ch = (o.get("choices") or [{}])[0]
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        content_parts.append(piece)
                        self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
            if not (prompt_tokens or comp_tokens):
                comp_tokens = est_tokens("".join(content_parts))
            record_usage(key, f"fallback:{FALLBACK_MODEL}", prompt_tokens, comp_tokens)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            with call_fallback_chat(model, messages, False, options) as r:
                o = json.loads(r.read().decode())
            u = o.get("usage") or {}
            content = (o.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            record_usage(key, f"fallback:{FALLBACK_MODEL}",
                         u.get("prompt_tokens") or est_tokens(json.dumps(messages)),
                         u.get("completion_tokens") or est_tokens(content))
            self._send(200, o)

def serve():
    port = int(os.environ.get("UNCORKED_PORT", "11500"))
    init_db()
    srv = ThreadingHTTPServer((HOST, port), H)
    print(f"UNCORKED gateway on http://{HOST}:{port}  (fallback={'on' if FALLBACK_BASE and FALLBACK_KEY else 'off'})", flush=True)
    srv.serve_forever()

# ---------- CLI ----------

def cmd_issue(a):
    import secrets
    tier = a.tier.lower()
    if tier not in TIER_BUDGET:
        raise SystemExit(f"tier must be one of {list(TIER_BUDGET)}")
    key = "uk_" + secrets.token_hex(12)
    with DB_LOCK:
        db().execute("INSERT INTO keys(key,name,tier,active,created) VALUES(?,?,?,1,?)",
                     (key, a.name, tier, now_iso()))
        db().commit()
    print(f"{key}  ({a.name}, {tier})")

def cmd_list(_):
    for r in db().execute("SELECT name,tier,active,created FROM keys ORDER BY created"):
        print(f"{r['created']}  {r['name']:<20} {r['tier']:<7} active={r['active']}")
    k = db().execute("SELECT key FROM keys").fetchall()
    if k: print("\nkeys stored in", DB_PATH)

def cmd_usage(a):
    rows = db().execute("SELECT key,name,tier FROM keys").fetchall() if not a.key else \
           [db().execute("SELECT key,name,tier FROM keys WHERE key=?", (a.key,)).fetchone()]
    for r in rows:
        if r is None: continue
        budget = TIER_BUDGET.get(r["tier"])
        used, used_raw, credits = usage_month(r["key"])
        line = f"{r['name']:<20} {r['tier']:<7} used={used:>10,}"
        if budget: line += f"  budget={budget:>10,}  credits={credits:>10,}"
        print(line + f"   ({used_raw:,} raw)")

def cmd_grant(a):
    with DB_LOCK:
        db().execute("INSERT INTO credits(key,ts,tokens,note) VALUES(?,?,?,?)",
                     (a.key, now_iso(), int(a.tokens), a.note))
        db().commit()
    print(f"granted {a.tokens:,} tokens credit to {a.key} ({a.note})")

def cmd_pack(a):
    with DB_LOCK:
        db().execute("INSERT INTO packs(key,ts,plan,usd,method) VALUES(?,?,?,?,?)",
                     (a.key, now_iso(), a.plan, float(a.usd), a.method))
        db().commit()
    print(f"pack recorded: {a.key} {a.plan} ${a.usd} via {a.method}")

def cmd_revoke(a):
    with DB_LOCK:
        db().execute("UPDATE keys SET active=0 WHERE key=?", (a.key,))
        db().commit()
    print("revoked")

def cmd_ledger(a):
    import csv, sys
    where = f"substr(ts,1,7)=?" if a.month else "1=1"
    args = (a.month,) if a.month else ()
    w = csv.writer(sys.stdout)
    w.writerow(["month", "ts", "key", "model", "prompt_tokens", "comp_tokens"])
    for r in db().execute(f"SELECT ts,key,model,prompt_tokens,comp_tokens FROM ledger WHERE {where} ORDER BY ts", args):
        w.writerow([r["ts"][:7], r["ts"], r["key"], r["model"], r["prompt_tokens"], r["comp_tokens"]])
    w.writerow([])
    w.writerow(["month", "ts", "key", "plan", "usd", "method"])
    for r in db().execute(f"SELECT ts,key,plan,usd,method FROM packs WHERE {where} ORDER BY ts", args):
        w.writerow([r["ts"][:7], r["ts"], r["key"], r["plan"], r["usd"], r["method"]])

def main():
    p = argparse.ArgumentParser(description="UNCORKED gateway")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("server")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    i = sub.add_parser("issue")
    i.add_argument("name")
    i.add_argument("tier")
    sub.add_parser("list")
    u = sub.add_parser("usage")
    u.add_argument("--key", nargs="?")
    u.add_argument("--month", default=None)
    g = sub.add_parser("grant")
    g.add_argument("key")
    g.add_argument("tokens", type=int)
    g.add_argument("--note", default="")
    pk = sub.add_parser("pack")
    pk.add_argument("key")
    pk.add_argument("plan")
    pk.add_argument("usd", type=float)
    pk.add_argument("method")
    r = sub.add_parser("revoke")
    r.add_argument("key")
    l = sub.add_parser("ledger")
    l.add_argument("--month", default=None)
    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return
    init_db()
    if args.cmd == "server":
        if args.host:
            os.environ["UNCORKED_HOST"] = args.host
        if args.port:
            os.environ["UNCORKED_PORT"] = str(args.port)
        serve()
    elif args.cmd == "issue":
        cmd_issue(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "usage":
        cmd_usage(args)
    elif args.cmd == "grant":
        cmd_grant(args)
    elif args.cmd == "pack":
        cmd_pack(args)
    elif args.cmd == "revoke":
        cmd_revoke(args)
    elif args.cmd == "ledger":
        cmd_ledger(args)

if __name__ == "__main__":
    main()
