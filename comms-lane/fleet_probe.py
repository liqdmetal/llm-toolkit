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
from urllib.parse import urlsplit

# (label, base_url, kind). kind decides which listing endpoint answers.
#
# q38 is a llama.cpp server on the SAME HOST as the 5090's ollama daemon, and it
# is a separate lane on purpose: measured 2026-11-26, the 5090's /api/ps answered
# {"models":[]} (ollama idle) while http://192.168.1.232:8081/health answered 200
# with `Server: llama.cpp` and a model loaded. Asking only the ollama question
# reported that box as IDLE while it was serving -- the exact fall-through to a
# billed lane this tool exists to catch.
LANES = [
    ("nous", "http://127.0.0.1:8099", "openai"),
    ("5090", "http://192.168.1.232:11434", "ollama"),
    ("q38", "http://192.168.1.232:8081", "llamacpp"),
    ("3090", "http://192.168.1.124:11434", "ollama"),
    # The 3090 runs TWO servers, exactly like the 5090 does, and asking only its
    # ollama daemon read the box as idle while llama.cpp served Inkling on it.
    # MEASURED 2026-11-28: /api/ps answered {"models":[]} (so "IDLE 0/7") and a
    # 3-token /api/generate died in 4.0 s with "cudaMemGetInfo failed (out of
    # memory)" because Inkling held the card, while the tunnel below --
    # ssh -L 18082:127.0.0.1:8082 to the 3090 -- answered {"status":"ok"} to
    # /health. Reporting a serving lane as idle is the fall-through to a billed
    # lane this tool exists to prevent, and it is the same defect the 5090 got
    # fixed for in test_fleet_llamacpp_lane.py, one box over.
    ("3090-inkling", "http://127.0.0.1:18082", "llamacpp"),
]

# A lane reached through an SSH tunnel answers on LOOPBACK, but the machine it
# measures is the FAR END. Without this map the 3090 is counted twice -- once
# serving (as 127.0.0.1) and once idle-with-7-installed (as 192.168.1.124) --
# and the two readings disagree about the same card in the expensive direction:
# the idle half invites a shaping dispatch onto a card that then fails to load.
# Attribute a tunnel to its far end; that is the only attribution under which
# "one host = one GPU" is true.
TUNNEL_HOSTS = {
    "http://127.0.0.1:18082": "192.168.1.124",
}

TIMEOUT = 6.0


def _get(url, timeout=TIMEOUT):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _status_and_json(url, timeout=TIMEOUT):
    """Return (http_status, parsed_json_or_None); never raise on 4xx/5xx.

    `_get` raises through urllib on an HTTP error, which is right for a listing
    endpoint that must deliver data. llama.cpp's readiness signal is the
    opposite: a 503 IS the answer (model still loading), so the status has to be
    readable rather than exceptional.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        return e.code, None
    try:
        return status, json.loads(raw.decode())
    except ValueError:
        return status, None


def list_models(base_url, kind, timeout=TIMEOUT):
    """The model names a lane reports. Raises on anything that is not a listing.

    Returns None where the lane CANNOT answer the question, which is different
    from [] (it answered, nothing installed). llama.cpp's /v1/models is key-gated
    on this fleet and answers 401, so a count of 0 there would be a fabricated
    absence -- the same defect this probe exists to catch, one field over.
    """
    base = base_url.rstrip("/")
    if kind == "ollama":
        doc = _get(base + "/api/tags", timeout)
        return [m["name"] for m in doc.get("models", [])]
    if kind == "openai":
        doc = _get(base + "/v1/models", timeout)
        return [m["id"] for m in doc.get("data", [])]
    if kind == "llamacpp":
        status, doc = _status_and_json(base + "/v1/models", timeout)
        if status in (401, 403):
            return None  # key-gated: unknown, never "none installed"
        if status != 200:
            raise ValueError("llamacpp /v1/models answered HTTP %s" % status)
        return [m["id"] for m in (doc or {}).get("data", [])]
    raise ValueError("unknown lane kind: %r" % (kind,))


def list_resident(base_url, kind, timeout=TIMEOUT):
    """Models the lane currently has LOADED and able to answer right now.

    Returns a list, or None where the question does not apply. Ollama answers
    /api/ps; a hosted OpenAI-shaped portal has no equivalent notion, so it gets
    None ("not applicable") rather than [] ("nothing loaded") -- the two must
    not be confused, which is the whole point of this function.

    /api/ps is a listing endpoint like /api/tags: it reads state and loads
    nothing, so probing cannot evict the model it is measuring.

    llama.cpp answers /health instead: 200 when a model is loaded and serving,
    503 during its load window. That is a readiness signal, not a name list, and
    the name is not recoverable without the key -- so a ready server reports ONE
    resident whose name is marked unknown rather than inventing one.
    """
    base = base_url.rstrip("/")
    if kind == "ollama":
        doc = _get(base + "/api/ps", timeout)
        return [m.get("name") or m.get("model") or "?"
                for m in doc.get("models", [])]
    if kind == "llamacpp":
        status, _doc = _status_and_json(base + "/health", timeout)
        if status == 200:
            # Ready. The model NAME is behind the key gate, so it is stated as
            # unknown instead of guessed.
            return ["<loaded; name key-gated>"]
        if status == 503:
            return []  # loading: up, but no capacity yet
        raise ValueError("llamacpp /health answered HTTP %s" % status)
    return None


def _host_of(url):
    """The MACHINE a lane lives on, not the daemon it reaches.

    Two daemons on one host are one GPU. Measured 2026-11-26: the 5090's ollama
    (”:11434”) and q38's llama.cpp (":8081") are both 192.168.1.232, so
    counting daemons as capacity described one loaded card as two idle lanes.
    """
    key = url.rstrip("/")
    if key in TUNNEL_HOSTS:
        return TUNNEL_HOSTS[key]
    try:
        return urlsplit(url).hostname or url
    except ValueError:
        return url


def host_capacity(rows):
    """Idle capacity per MACHINE, and the daemons whose idleness is misleading.

    A lane's ``serving`` flag answers "is this daemon holding a loaded model".
    Capacity answers "is this GPU free". They come apart whenever one host runs
    two inference servers, and they come apart in the expensive direction: an
    idle ollama daemon on a host whose llama.cpp server has a model loaded reads
    as spare capacity and invites a dispatch onto a busy card.

    Returns (hosts, free, shared, unread_hosts):
      hosts        every host with at least one local daemon
      free         hosts where nothing is serving and at least one daemon answered
      shared       lanes that are idle on a host that is already serving
      unread_hosts hosts where no daemon could answer residency at all
    """
    by_host = {}
    for r in rows:
        if r["resident"] is None and r["resident_error"] is None:
            continue  # hosted: no residency notion, not part of local capacity
        by_host.setdefault(_host_of(r["url"]), []).append(r)
    free, shared, unread_hosts = [], [], []
    for host, lanes in by_host.items():
        busy = any(x["serving"] for x in lanes)
        answered = any(x["resident"] is not None for x in lanes)
        if not busy and answered:
            free.append(host)
        if not answered:
            unread_hosts.append(host)
        for x in lanes:
            if busy and x["resident"] is not None and not x["serving"]:
                shared.append(x["lane"])
    return by_host, free, shared, unread_hosts


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
            "model_count": None if models is None else len(models),
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
        # model_count is None where the lane cannot answer the listing question
        # (a key-gated /v1/models). Rendering that as 0 would report a
        # fabricated absence, which is the defect this probe exists to catch.
        count = "? installed" if r["model_count"] is None \
            else "%d installed" % r["model_count"]
        if r["resident"] is None:
            serving = "hosted"
        elif r["serving"]:
            serving = "serving %d" % r["resident_count"]
        else:
            serving = "IDLE 0/%s loaded" % (
                "?" if r["model_count"] is None else r["model_count"])
        print("  %-*s UP    %s, %s, %d ms"
              % (width, r["lane"], count, serving, r["latency_ms"]))
        if r["resident_error"]:
            print("  %-*s       residency unknown: %s"
                  % (width, r["lane"], r["resident_error"]))
    up = sum(1 for r in rows if r["reachable"])
    print("[%d/%d] lanes reachable" % (up, len(rows)))
    # The number that was invisible before: local capacity that is up and
    # doing nothing. A reachable-but-idle lane is the expensive default this
    # tool exists to expose, because the router falls through to a billed one.
    #
    # `resident` is None for TWO different situations and they must not be
    # filtered together: a hosted lane has no residency notion at all, while a
    # LOCAL lane whose /api/ps failed is a lane that could not be asked. Building
    # the denominator from `resident is not None` dropped the second kind, so a
    # local lane that answered /api/ps and then failed rendered as if it were
    # not in the fleet -- the count got SMALLER and HEALTHIER than the truth.
    local = [r for r in rows
             if r["resident"] is not None or r["resident_error"] is not None]
    idle = [r for r in local if r["resident"] is not None and not r["serving"]]
    unread = [r for r in local if r["resident"] is None]
    if local:
        detail = ", ".join(r["lane"] for r in idle) or "none"
        if unread:
            detail += "; residency unread: " + ", ".join(
                r["lane"] for r in unread)
        print("[%d/%d] local lanes SERVING (idle: %s)"
              % (len(local) - len(idle) - len(unread), len(local), detail))
        # The lane count above is DAEMONS. Capacity is MACHINES, and the two
        # differ whenever one host runs more than one inference server --
        # measured 2026-11-26, the 5090's ollama and q38's llama.cpp are both
        # 192.168.1.232, so an idle ollama daemon beside a loaded llama.cpp
        # server rendered as spare capacity on a card that was already busy.
        # That reading is the expensive direction: it invites a dispatch onto
        # a busy GPU. So state the machine count too, and name the lanes whose
        # idleness is an artifact of which daemon was asked.
        by_host, free, shared, unread_hosts = host_capacity(rows)
        if by_host:
            host_word = "host" if len(by_host) == 1 else "hosts"
            # NOT "idle capacity". This probe deliberately never starts a
            # generation, so the strongest fact it can establish is that no
            # daemon on the host is SERVING right now. Capacity is a
            # different claim, and they came apart in the expensive
            # direction: MEASURED 2026-11-26, the 3090 answered
            # {"models":[]} to /api/ps while every generation on it died
            # "cudaMemGetInfo failed (out of memory)" because inkling held
            # the card. This line called that box "[1/2] idle capacity", and
            # a router reading it dispatches into a host that cannot load,
            # gets nothing back, and falls through to the billed lane this
            # tool exists to keep it off. State what was measured and name
            # the limit beside it.
            line = ("[%d/%d] local %s with no daemon serving "
                    "(capacity NOT established: a listed model can still "
                    "fail to load)") % (len(free), len(by_host), host_word)
            notes = []
            if shared:
                notes.append("idle daemons on a busy host: " + ", ".join(shared))
            if unread_hosts:
                notes.append("residency unread on: " + ", ".join(
                    sorted(unread_hosts)))
            if notes:
                line += " (" + "; ".join(notes) + ")"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
