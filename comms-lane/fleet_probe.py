"""fleet_probe.py — one row per lane, measured rather than assumed.

Why this exists: ROUTING.md routes work by lane, but nothing in this repo could
answer "is that lane actually up, and does it have the model the policy names?".
Both failures look identical from the router's side — work goes out, nothing
comes back — and both end with the same expensive default: a billed frontier
lane. The probe turns that into a number per lane.

Measured 2026-11-24, which is the first thing this tool established:

    5090 ollama (192.168.1.232:11434)   3 models: 7b-qwen-q4, qwen2.5:7b,
                                                  llama3.2-vision
    3090 ollama (192.168.1.124:11434)   7 models incl. defiant9b-abliterated:256k
    nous portal (127.0.0.1:8099)      410 models, 0.27 s

ROUTING.md rule 1 routes "reasoning-heavy / sensitive" work to "q38 jumbo,
defiant abliterated, Qwen heretic on the 5090". Two of those three are not on
the box it names, so that row cannot execute as written. This tool is what makes
that statement checkable instead of a doc that drifts.

Deliberately cheap: /api/tags and /v1/models list installed models and load
none. Probing must never evict a resident model or start a generation, because a
probe that perturbs what it measures is worse than no probe.

Stdlib only. Usage:
    python fleet_probe.py            # one row per lane
    python fleet_probe.py --json     # machine-readable
Exit 0 whether lanes are up or down: a down lane is data, not an error of the
probe. Only a broken probe exits non-zero.
"""
import json
import sys
import time
import urllib.error
import urllib.request

# (label, base_url, kind). kind decides which listing endpoint answers.
LANES = [
    ("nous", "http://127.0.0.1:8099", "openai"),
    ("5090", "http://192.168.1.232:11434", "ollama"),
    ("3090", "http://192.168.1.124:11434", "ollama"),
]

TIMEOUT = 6.0


def _get(url, timeout=TIMEOUT):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def list_models(base_url, kind, timeout=TIMEOUT):
    """The model names a lane reports. Raises on anything that is not a listing."""
    if kind == "ollama":
        doc = _get(base_url.rstrip("/") + "/api/tags", timeout)
        return [m["name"] for m in doc.get("models", [])]
    if kind == "openai":
        doc = _get(base_url.rstrip("/") + "/v1/models", timeout)
        return [m["id"] for m in doc.get("data", [])]
    raise ValueError("unknown lane kind: %r" % (kind,))


def list_resident(base_url, kind, timeout=TIMEOUT):
    """Models the lane currently has LOADED and able to answer right now.

    Returns a list, or None where the question does not apply. Ollama answers
    /api/ps; a hosted OpenAI-shaped portal has no equivalent notion, so it gets
    None ("not applicable") rather than [] ("nothing loaded") -- the two must
    not be confused, which is the whole point of this function.

    /api/ps is a listing endpoint like /api/tags: it reads state and loads
    nothing, so probing cannot evict the model it is measuring.
    """
    if kind != "ollama":
        return None
    doc = _get(base_url.rstrip("/") + "/api/ps", timeout)
    return [m.get("name") or m.get("model") or "?" for m in doc.get("models", [])]


def probe(label, base_url, kind, timeout=TIMEOUT):
    """One lane's state. Never raises: a failure IS the measurement."""
    started = time.time()
    try:
        models = list_models(base_url, kind, timeout)
        # Reachable says the process answers; residency says it can serve now.
        # A lane with 7 models installed and none loaded is UP and IDLE, and
        # reporting only "7 models" made those two states identical.
        try:
            resident = list_resident(base_url, kind, timeout)
            resident_error = None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError) as exc:
            resident, resident_error = None, "%s: %s" % (type(exc).__name__, exc)
        return {
            "lane": label,
            "url": base_url,
            "reachable": True,
            "models": models,
            "model_count": len(models),
            "resident": resident,
            "resident_count": None if resident is None else len(resident),
            "serving": None if resident is None else bool(resident),
            "resident_error": resident_error,
            "latency_ms": round((time.time() - started) * 1000),
            "error": None,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, KeyError) as exc:
        return {
            "lane": label,
            "url": base_url,
            "reachable": False,
            "models": [],
            "model_count": 0,
            "resident": None,
            "resident_count": None,
            "serving": None,
            "resident_error": None,
            "latency_ms": round((time.time() - started) * 1000),
            "error": "%s: %s" % (type(exc).__name__, exc),
        }


def probe_all(lanes=LANES, timeout=TIMEOUT):
    return [probe(label, url, kind, timeout) for label, url, kind in lanes]


def main(argv):
    as_json = "--json" in argv
    rows = probe_all()
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    width = max(len(r["lane"]) for r in rows)
    for r in rows:
        if not r["reachable"]:
            print("  %-*s DOWN  %s" % (width, r["lane"], r["error"]))
            continue
        if r["resident"] is None:
            serving = "hosted"
        elif r["serving"]:
            serving = "serving %d" % r["resident_count"]
        else:
            serving = "IDLE 0/%d loaded" % r["model_count"]
        print("  %-*s UP    %d installed, %s, %d ms"
              % (width, r["lane"], r["model_count"], serving, r["latency_ms"]))
        if r["resident_error"]:
            print("  %-*s       residency unknown: %s"
                  % (width, r["lane"], r["resident_error"]))
    up = sum(1 for r in rows if r["reachable"])
    print("[%d/%d] lanes reachable" % (up, len(rows)))
    # The number that was invisible before: local capacity that is up and
    # doing nothing. A reachable-but-idle lane is the expensive default this
    # tool exists to expose, because the router falls through to a billed one.
    local = [r for r in rows if r["resident"] is not None]
    idle = [r for r in local if not r["serving"]]
    if local:
        print("[%d/%d] local lanes SERVING (idle: %s)"
              % (len(local) - len(idle), len(local),
                 ", ".join(r["lane"] for r in idle) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
