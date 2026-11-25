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
import os
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
    # The 3090 runs a THIRD server and it was invisible to this roster entirely.
    # MEASURED 2026-11-28: loopback 8090 is an ssh tunnel --
    #   ssh -fN -i ...\a3090_ed25519 -L 8090:127.0.0.1:8090 a3090@192.168.1.124
    # (PID 7824, netstat) -- and it answers keyless with a model loaded:
    #   /health 200 {"status":"ok"}, /v1/models 200 -> /home/a3090/mm/defiant-vision.gguf,
    #   /slots 4 slots, 0 busy, n_ctx 4096 each.
    # So the same box printed "3090 UP 7 installed, IDLE 0/7 loaded" while a
    # llama.cpp server on it held defiant-vision. Same defect as the 5090's
    # ollama-beside-llama.cpp (2026-11-26) and the 3090's ollama-beside-Inkling
    # (2026-11-28), a third daemon later: asking one kind of daemon and printing
    # the answer as the box's state.
    ("3090-defiant", "http://127.0.0.1:8090", "llamacpp"),
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
    # Its sibling on the same far end. Both tunnels answer on loopback and both
    # measure the 3090; without this the box is counted three times -- idle on
    # its ollama daemon, serving Inkling, serving defiant-vision -- and the
    # idle third is what a router would act on.
    "http://127.0.0.1:8090": "192.168.1.124",
}

TIMEOUT = 6.0

# Optional per-lane API key. Some llama.cpp servers on this fleet key-gate
# BOTH /v1/models and /slots and answer 401 without one. MEASURED 2026-11-29 by
# reading all three local lanes keyless:
#
#     q38           /health 200   /v1/models 401   /slots 401
#     3090-inkling  /health 200   /v1/models 401   /slots 401
#     3090-defiant  /health 200   /v1/models 200   /slots 200
#
# So /health is the only keyless read, and it answers exactly one bit: a model
# is loaded. Spare capacity lives in /slots, which two of the three loaded
# servers will not surrender without a key. Before this, `fleet_probe` had no
# way to send one, so those two lanes printed `slots UNKNOWN` on every run
# forever -- capacity that demonstrably exists and is never measured, while the
# router reads unknown as "do not dispatch there" and falls through to a billed
# lane. That is the exact expensive default this tool exists to prevent.
#
# The key is read from the environment and never stored in the row. The row
# records WHICH variable supplied it (`key_source`), never its value, because a
# report that prints a secret to a terminal and into --json is a leak that
# outlives the run.
LANE_KEY_ENV = "FLEET_LANE_API_KEY"          # fallback for every lane
LANE_KEY_ENV_PREFIX = "FLEET_LANE_API_KEY_"   # per-lane override


def key_env_name(label):
    """The environment variable that may hold an API key for this lane."""
    slug = "".join(c if c.isalnum() else "_" for c in label).upper()
    return LANE_KEY_ENV_PREFIX + slug


def key_for(label):
    """(key, source_var) for a lane; (None, None) when nothing is configured.

    Per-lane first, then the shared fallback. Returns the variable NAME as well
    as the value so a caller can say which knob to turn without ever printing
    the secret itself.
    """
    for name in (key_env_name(label), LANE_KEY_ENV):
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip(), name
    return None, None


def _request(url, timeout=TIMEOUT, key=None):
    """An open urlopen response, carrying the key when one was configured."""
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    return urllib.request.urlopen(req, timeout=timeout)


def _get(url, timeout=TIMEOUT, key=None):
    with _request(url, timeout, key) as r:
        return json.loads(r.read().decode())


def _status_and_json(url, timeout=TIMEOUT, key=None):
    """Return (http_status, parsed_json_or_None); never raise on 4xx/5xx.

    `_get` raises through urllib on an HTTP error, which is right for a listing
    endpoint that must deliver data. llama.cpp's readiness signal is the
    opposite: a 503 IS the answer (model still loading), so the status has to be
    readable rather than exceptional.
    """
    try:
        with _request(url, timeout, key) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        return e.code, None
    try:
        return status, json.loads(raw.decode())
    except ValueError:
        return status, None


def list_models(base_url, kind, timeout=TIMEOUT, key=None):
    """The model names a lane reports. Raises on anything that is not a listing.

    Returns None where the lane CANNOT answer the question, which is different
    from [] (it answered, nothing installed). llama.cpp's /v1/models is key-gated
    on this fleet and answers 401, so a count of 0 there would be a fabricated
    absence -- the same defect this probe exists to catch, one field over.
    """
    base = base_url.rstrip("/")
    if kind == "ollama":
        doc = _get(base + "/api/tags", timeout, key)
        return [m["name"] for m in doc.get("models", [])]
    if kind == "openai":
        doc = _get(base + "/v1/models", timeout, key)
        return [m["id"] for m in doc.get("data", [])]
    if kind == "llamacpp":
        status, doc = _status_and_json(base + "/v1/models", timeout, key)
        if status in (401, 403):
            return None  # key-gated: unknown, never "none installed"
        if status != 200:
            raise ValueError("llamacpp /v1/models answered HTTP %s" % status)
        return [m["id"] for m in (doc or {}).get("data", [])]
    raise ValueError("unknown lane kind: %r" % (kind,))


def list_resident(base_url, kind, timeout=TIMEOUT, key=None):
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
        doc = _get(base + "/api/ps", timeout, key)
        return [m.get("name") or m.get("model") or "?"
                for m in doc.get("models", [])]
    if kind == "llamacpp":
        status, _doc = _status_and_json(base + "/health", timeout, key)
        if status == 200:
            # Ready. The model NAME is behind the key gate, so it is stated as
            # unknown instead of guessed.
            return ["<loaded; name key-gated>"]
        if status == 503:
            return []  # loading: up, but no capacity yet
        raise ValueError("llamacpp /health answered HTTP %s" % status)
    return None


def read_slots(base_url, kind, timeout=TIMEOUT, key=None):
    """(total_slots, busy_slots) for a llama.cpp server; None where N/A.

    Residency answers "is a model loaded", which is what ``serving`` means. It
    does NOT answer "is this server busy right now": a loaded server with every
    slot free renders identically to a saturated one. MEASURED 2026-11-28 --
    the defiant-vision tunnel answered ``/slots`` with 4 slots, 0 busy, while
    this probe printed ``serving 1`` and nothing else. That is spare capacity
    the fleet report could not see, which is the expensive direction for the
    goal this tool serves: work gets dispatched to a billed lane because the
    local one looked as busy as it can.

    ``/slots`` is a STATE READ. It lists the server's in-flight work and starts
    none, so it satisfies the standing rule that a probe must not perturb what
    it measures. It is also KEY-GATED on some boxes: measured 2026-11-28, q38
    (127.0.0.1:8081 by way of the .232 host) answered 401 and the Inkling
    tunnel (18082) answered 401, while the defiant-vision tunnel (8090)
    answered. A 401 is NOT "no slots" and NOT "0 busy" -- it is "could not
    ask", and the caller must render it as unknown rather than as a measured
    zero. Raising is how this function says so; the caller decides the wording.
    """
    if kind != "llamacpp":
        return None
    base = base_url.rstrip("/")
    doc = _get(base + "/slots", timeout, key)
    if not isinstance(doc, list):
        raise ValueError("/slots answered %s, not a list" % type(doc).__name__)
    busy = 0
    for slot in doc:
        # A slot mid-prefill or mid-decode both count as busy; only a truly
        # free slot is capacity.
        if isinstance(slot, dict) and (
                slot.get("is_processing") or slot.get("is_processing_prompt")):
            busy += 1
    return len(doc), busy


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


def probe(label, base_url, kind, timeout=TIMEOUT, key=None):
    """One lane's state. Never raises: a failure IS the measurement.

    ``key`` is optional and, when given, is sent as a bearer token on every
    read. The row records the NAME of the variable that supplied it and never
    the value, so a report can say "this lane was read with a key" without
    writing a secret into a terminal scrollback or a --json file.
    """
    started = time.time()
    key_src = key_for(label)[1] if key else None
    try:
        models = list_models(base_url, kind, timeout, key)
        # Reachable says the process answers; residency says it can serve now.
        # A lane with 7 models installed and none loaded is UP and IDLE, and
        # reporting only "7 models" made those two states identical.
        try:
            resident = list_resident(base_url, kind, timeout, key)
            resident_error = None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError) as exc:
            resident, resident_error = None, "%s: %s" % (type(exc).__name__, exc)
        try:
            slots = read_slots(base_url, kind, timeout, key)
            slots_error = None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError) as exc:
            slots, slots_error = None, "%s: %s" % (type(exc).__name__, exc)
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
            "slots": slots,
            "slots_error": slots_error,
            "key_source": key_src,
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
            "slots": None,
            "slots_error": None,
            "key_source": key_src,
            "latency_ms": round((time.time() - started) * 1000),
            "error": "%s: %s" % (type(exc).__name__, exc),
        }


def probe_all(lanes=LANES, timeout=TIMEOUT):
    """One row per lane, resolving each lane's own key (or none).

    The key is resolved per lane here rather than passed down from main(), so a
    caller that builds its own lane list still gets the same behaviour, and so
    the environment is read at probe time instead of at import.
    """
    return [probe(label, url, kind, timeout, key_for(label)[0])
            for label, url, kind in lanes]


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
            # "serving" says a model is LOADED. It does not say the server is
            # busy, and those two came apart in the direction that costs money:
            # a loaded server with every slot free looked exactly like a
            # saturated one. State the slots when they were read; when the read
            # was refused (key-gated) say UNKNOWN rather than print nothing,
            # because silence there reads as "no spare capacity".
            if r.get("slots") is not None:
                slots_total, slots_busy = r["slots"]
                serving += ", %d/%d slots busy" % (slots_busy, slots_total)
            elif r.get("slots_error"):
                serving += ", slots UNKNOWN"
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
        # MEASURED spare capacity, which is not the same claim as the host line
        # below. A loaded llama.cpp server whose /slots answered 0 busy is a
        # server that CAN take work right now -- the model is loaded, so it
        # cannot fail to load the way the host line warns about. Only lanes
        # whose /slots was actually READ appear here; a key-gated lane is not
        # counted as full, and not counted as free either.
        spare = [r for r in local
                 if r.get("slots") is not None and r["slots"][0] > 0
                 and r["slots"][1] == 0]
        if spare:
            print("[%d] loaded local server(s) with every slot free (MEASURED "
                  "spare capacity: %s)"
                  % (len(spare), ", ".join(
                      "%s 0/%d" % (r["lane"], r["slots"][0]) for r in spare)))
        gated = [r for r in local
                 if r.get("slots") is None and r.get("slots_error")
                 and r.get("serving")]
        if gated:
            # Name the KNOB, not just the gap. MEASURED 2026-11-29: q38 and the
            # Inkling tunnel both answer 401 on /v1/models AND /slots, so this
            # line printed on every run for two loaded servers forever. A
            # report that says "unknown" and not "unknown because no key was
            # configured, set FLEET_LANE_API_KEY_Q38" is a report that never
            # tells the operator the one action that would turn it into a
            # measurement.
            # THREE states here, not two. MEASURED 2026-11-29: with NO key
            # configured the line blamed the missing key; with a WRONG key
            # configured it said nothing at all, so "you never set the
            # variable" and "you set it and the server rejected it" -- the
            # two failures an operator must distinguish, because one is fixed
            # by setting a variable and the other by correcting its value --
            # rendered identically. The second is the worse one to be silent
            # about: the operator has done the thing the report asked and the
            # report stops talking, which reads as success.
            unnamed = [r["lane"] for r in gated if not r.get("key_source")]
            refused = [r for r in gated if r.get("key_source")]
            hints = []
            if unnamed:
                hints.append("no API key configured; set %s or %s"
                             % (key_env_name(unnamed[0]), LANE_KEY_ENV))
            for r in refused:
                hints.append("%s sent %s and the server refused it"
                             % (r["lane"], r["key_source"]))
            hint = (" -- " + "; ".join(hints)) if hints else ""
            print("[%d] serving local lane(s) whose busy count could not be "
                  "read (not counted as free): %s%s"
                  % (len(gated), ", ".join(r["lane"] for r in gated), hint))
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
