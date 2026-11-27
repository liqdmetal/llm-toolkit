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


# Why a `/slots` read failed, as distinct CAUSES. The report named one cause
# for all of them and it was the wrong one in most arms. MEASURED 2026-12-03 by
# driving the shipped `main()` over six error texts on one gated lane:
#
#     401 Unauthorized        -> "no API key configured; set FLEET_LANE_API_KEY_Q38"
#     timeout                 -> the same sentence
#     500 Internal Server Err -> the same sentence
#     403 Forbidden           -> the same sentence
#     200 but not a slot list -> the same sentence
#     404 Not Found           -> the same sentence
#
# A key cannot fix a timeout, a 500, a missing endpoint or a payload of the
# wrong SHAPE, and an operator who sets a variable because the report told them
# to, reruns, and gets the identical sentence has been sent to fix a thing that
# was never wrong. The report exists to say which action turns an unread figure
# into a measurement, so it has to name the cause it ACTUALLY saw.
SLOTS_CAUSE_AUTH = "auth"
SLOTS_CAUSE_TIMEOUT = "timeout"
SLOTS_CAUSE_HTTP = "http"
SLOTS_CAUSE_SHAPE = "shape"
SLOTS_CAUSE_UNREACHABLE = "unreachable"
SLOTS_CAUSE_UNKNOWN = "unreadable"


def slots_cause(err):
    """Which cause a recorded `/slots` failure names.

    Classified from the text `probe()` records (`"<Type>: <message>"`), because
    the row shape is what every existing caller and test fixture speaks. An
    unrecognised failure is `unreadable`, which claims nothing.
    """
    t = err or ""
    if "401" in t or "403" in t or "Unauthorized" in t or "Forbidden" in t:
        return SLOTS_CAUSE_AUTH
    if "timed out" in t or "timeout" in t.lower():
        return SLOTS_CAUSE_TIMEOUT
    if "HTTP Error" in t or "HTTPError" in t:
        return SLOTS_CAUSE_HTTP
    if "not a list" in t or "ValueError" in t or "KeyError" in t:
        return SLOTS_CAUSE_SHAPE
    if "URLError" in t or "OSError" in t or "Connection" in t:
        return SLOTS_CAUSE_UNREACHABLE
    return SLOTS_CAUSE_UNKNOWN


def key_env_name(label):
    """The environment variable that may hold an API key for this lane."""
    slug = "".join(c if c.isalnum() else "_" for c in label).upper()
    return LANE_KEY_ENV_PREFIX + slug


HOSTED_KINDS = ("openai",)
"""Kinds that are a HOSTED portal rather than a local daemon.

A hosted portal has no residency notion at all: there is no /api/ps to answer,
so "is a model loaded on it" is not a question that applies. A LOCAL daemon that
could not be reached reaches the same all-None row shape for the opposite
reason. `reachable` and this tuple are what keep the two apart.
"""


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
    slots, _ctx, _n = read_slot_state(base_url, kind, timeout, key)
    return slots


def read_slot_detail(base_url, kind, timeout=TIMEOUT, key=None):
    """One `/slots` read -> every field the report needs, from ONE read.

    Added because the report needs a fourth fact (`parked`) and the module's own
    rule forbids getting it with a second read: "probing must never perturb what
    it measures, and two reads of a live server can disagree with each other".
    So this carries the three existing fields plus the new one out of the same
    single HTTP GET, and `read_slot_state` stays a thin wrapper over it -- its
    tuple shape is unchanged, which is what keeps the dozen existing callers and
    their tests honest.

    `parked` is the slots whose busy flag is CLEAR but which still hold a task:
    an `id_task` the server has assigned, and prompt tokens it has ingested but
    not advanced. MEASURED 2026-12-04 against the live fleet, nine reads over ~40
    seconds of `3090-defiant` (127.0.0.1:8090), every read identical:

        slot0-2  is_processing false, id_task None
        slot3    is_processing false, id_task 158, n_prompt_tokens 19,
                 n_prompt_tokens_processed 0

    while `fleet_probe` printed `3090-defiant 0/4 ..., MEASURED spare capacity`.
    That is the sharpest form of this repo's standing class: not a zero standing
    in for an absence, but a MAXIMUM standing in for an unmeasured slot. The
    busy flags say what a slot is doing; they do not say whether it is FREE, and
    a slot holding an assigned task is not evidence of freedom.

    It is deliberately NOT counted as busy. Whether that task is reserved work
    about to prefill or state left behind by a finished one cannot be decided
    from this document, and inventing either answer would be the same defect in
    the opposite direction. It is reported as its own named state, so the spare
    capacity figure and the fact that qualifies it travel together.

    `id_task` is compared with `>= 0` rather than tested for presence: a build
    that reports `-1` for an idle slot must not read as holding task 0.
    """
    if kind != "llamacpp":
        # Same N/A discipline as read_slot_state: an ollama lane has no /slots
        # notion at all, and that is not a failed read.
        return {"slots": None, "slot_ctx": None, "slot_ctx_slots": None,
                "parked": None}
    base = base_url.rstrip("/")
    doc = _get(base + "/slots", timeout, key)
    if not isinstance(doc, list):
        raise ValueError("/slots answered %s, not a list" % type(doc).__name__)
    busy = 0
    contexts = []
    parked = []
    for slot in doc:
        if not isinstance(slot, dict):
            continue
        processing = bool(slot.get("is_processing")
                          or slot.get("is_processing_prompt"))
        if processing:
            busy += 1
        n_ctx = slot.get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            contexts.append(n_ctx)
        if not processing:
            task = slot.get("id_task")
            if isinstance(task, int) and task >= 0:
                parked.append({"slot": slot.get("id"), "id_task": task,
                               "prompt_tokens": slot.get("n_prompt_tokens"),
                               "processed": slot.get(
                                   "n_prompt_tokens_processed")})
    return {"slots": (len(doc), busy),
            "slot_ctx": min(contexts) if contexts else None,
            "slot_ctx_slots": len(contexts),
            "parked": parked}


#: The smallest context any slot offers, or None when the server stated none.
#: Named so the render path has one place to ask instead of reading the row key
#: ad hoc in three of them.
def _ctx_suffix(row):
    """`", n_ctx <N>"` when this lane's context was MEASURED, else nothing.

    COVERAGE, not just the figure. MEASURED 2026-12-03, driving
    read_slot_state over five /slots documents: a 4-slot server where ONE slot
    stated n_ctx=4096 produced the identical suffix to a server where all four
    did -- `, n_ctx 4096`. The figure was a real reading; the sentence implied
    it described the whole server. llama.cpp assigns a request to whichever slot
    is free, so the context a caller can rely on is the least ANY slot offers,
    and a min taken over one of four slots is a claim about that slot only. A
    router told "n_ctx 4096" on a server whose other three slots were never
    described has been told the measurement's value and not its scope, which is
    the same defect class as the missing figure this function was written for:
    a real number reported without the fact that decides whether it can be used.

    A fully-covered server renders exactly as before, because that reading IS
    complete and adding words to it would be noise on the common case. A partly
    covered one says how many slots stated a context, so the number and its
    scope travel together.

    WHY THE SUMMARY LINES NEED THIS, measured 2026-12-02: `fleet_probe` printed

        [1] loaded local server(s) with every slot free (MEASURED spare
            capacity: 3090-defiant 0/4)

    and that one certified-free lane answered `/slots` with `n_ctx=4096` on all
    four slots, while `shape.py`'s default shaper model is
    `defiant9b-abliterated:256k` -- 262,144 tokens. So the ONLY lane in the
    fleet whose busy count is readable was certified as spare capacity while
    being 64x too small to hold the policy's own default prompt. A router that
    trusted the line would send the work, hit a context-exceeded failure, and
    fall through to the billed frontier lane -- the exact outcome this probe
    exists to prevent, arrived at THROUGH the probe's own certificate. The
    number was not wrong; it was incomplete.

    An unstated `n_ctx` prints nothing at all. Absent is not 4096 and not 0: a
    server that does not say must not be given a figure, or the field becomes a
    guess on the two lanes whose `/slots` is key-gated and on any server run
    with an older build that omits the key.
    """
    ctx = row.get("slot_ctx")
    if ctx is None:
        return ""
    covered = row.get("slot_ctx_slots")
    slots = row.get("slots")
    total = slots[0] if isinstance(slots, tuple) and slots else None
    if covered is None or total is None or covered >= total:
        # Fully covered, or the total is unknown -- in the second case there is
        # no denominator to state, and inventing one would be a fabricated
        # figure in the field that exists to stop fabricated figures.
        return f", n_ctx {ctx}"
    return f", n_ctx {ctx} ({covered} of {total} slots stated a context)"


def read_slot_state(base_url, kind, timeout=TIMEOUT, key=None):
    """One `/slots` read -> ``((total, busy), slot_ctx, ctx_slots)``.

    `read_slots` is kept as the `(total, busy)` shorthand every existing caller
    and test uses; this is the same single HTTP read with the context carried
    instead of discarded. One read, not two: probing must never perturb what it
    measures, and two reads of a live server can disagree with each other.

    `slot_ctx` is the SMALLEST `n_ctx` across slots, because llama.cpp assigns a
    request to whichever slot is free -- so the context a caller can actually
    rely on is the least any slot offers. Reporting the largest would certify
    capacity that a prompt can still fail to fit inside. The aggregate is a
    choice, and a silent choice is how a figure turns into an unfalsifiable
    claim, so the basis is stated here rather than left to the reader.

    `ctx_slots` is HOW MANY slots stated an `n_ctx` at all, because a minimum
    over one slot and a minimum over four are different facts and the tuple
    could not tell them apart. MEASURED 2026-12-03: one measured slot out of
    four yielded the same ``slot_ctx`` as four measured slots. The count is
    returned rather than folded into a boolean so a consumer can say "1 of 4"
    instead of only "partial".
    """
    if kind != "llamacpp":
        # THREE values, not two. `(total, busy)` is N/A for an ollama lane and
        # so is the coverage count -- but a two-value return in a
        # three-value function raises ValueError at the unpack, and probe()'s
        # except turns that into a slots_error: an ollama lane, whose /slots
        # question does not apply, would report as an attempted-and-failed
        # read. Caught by the both-trees control that pins N/A against error
        # (tests/test_fleet_slots_are_not_residency.py).
        return None, None, None
    detail = read_slot_detail(base_url, kind, timeout, key)
    return detail["slots"], detail["slot_ctx"], detail["slot_ctx_slots"]


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
        # A HOSTED lane has no residency notion (`resident` and
        # `resident_error` both None because the question does not apply). A
        # DOWN lane reaches the same two Nones for the opposite reason: nothing
        # was read because nothing answered. MEASURED 2026-12-03 at 297081a --
        # this one predicate collapsed both, so a local lane that never
        # responded was skipped as if it were hosted and its host appeared in
        # NO line of the report: not free, not shared, not unread.
        #
        # `reachable` is the field that separates them, and it is the right one:
        # it is False exactly when the probe could not get an answer, which is
        # the fact being reported.
        # A HOSTED portal is never local capacity, whether it answered or not.
        # Checked BEFORE the unreachable branch: nous timing out is a hosted
        # lane in trouble, not a local daemon whose residence could not be
        # established, and reporting it as the latter would name a local host
        # that does not exist.
        if r.get("kind") in HOSTED_KINDS:
            continue
        if not r["reachable"]:
            # Not asked -> the host's capacity is UNKNOWN, which is the
            # `unread_hosts` bucket below, not silence.
            by_host.setdefault(_host_of(r["url"]), []).append(r)
            continue
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
            _detail = read_slot_detail(base_url, kind, timeout, key)
            slots = _detail["slots"]
            slot_ctx = _detail["slot_ctx"]
            slot_ctx_slots = _detail["slot_ctx_slots"]
            parked = _detail["parked"]
            slots_error = None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError) as exc:
            slots, slot_ctx, slot_ctx_slots, parked, slots_error = \
                None, None, None, None, "%s: %s" % (type(exc).__name__, exc)
        return {
            "lane": label,
            "url": base_url,
            # The lane KIND travels on the row so a consumer can tell a hosted
            # portal from a local daemon without re-reading LANES -- and so it
            # can do that on a DOWN lane, where residency says nothing.
            "kind": kind,
            "reachable": True,
            "models": models,
            "model_count": None if models is None else len(models),
            "resident": resident,
            "resident_count": None if resident is None else len(resident),
            "serving": None if resident is None else bool(resident),
            "resident_error": resident_error,
            "slots": slots,
            "slots_error": slots_error,
            # The cause travels WITH the row so the summary never has to
            # re-guess it from a prose sentence, and so --json carries it too.
            "slots_cause": (None if slots_error is None
                            else slots_cause(slots_error)),
            "slot_ctx": slot_ctx,
            # How many slots stated an n_ctx. The figure above is a min over
            # these, so 1 and 4 describe very different servers. None when
            # /slots was never read (a 401, a timeout) -- no read, no count.
            "slot_ctx_slots": slot_ctx_slots,
            # Slots whose busy flag is clear but which still HOLD a task. The
            # spare-capacity line used to certify this lane as free capacity
            # while one slot held an assigned task -- see read_slot_detail.
            "parked": parked,
            "key_source": key_src,
            "latency_ms": round((time.time() - started) * 1000),
            "error": None,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, KeyError) as exc:
        return {
            "lane": label,
            "url": base_url,
            "kind": kind,
            "reachable": False,
            # None, NOT []. A down lane could not answer the listing question at
            # all, so it does not know that nothing is installed. MEASURED
            # 2026-12-03 at 297081a: probe() against a closed port and probe()
            # against a live server answering {"models": []} produced
            # BYTE-IDENTICAL models/model_count -- both []/0 -- so `--json`, the
            # machine-readable surface, could not tell a dead lane from an empty
            # live one. The human renderer was safe only because it prints
            # r["error"] and `continue`s before touching these keys; the JSON
            # consumer has no such guard.
            # This is the contract list_models() already documents: "Returns
            # None where the lane CANNOT answer the question, which is different
            # from [] (it answered, nothing installed)". The failure path was
            # the one branch that broke the module's own stated rule.
            "models": None,
            "model_count": None,
            "resident": None,
            "resident_count": None,
            "serving": None,
            "resident_error": None,
            "slots": None,
            "slots_error": None,
            # Carried on the down row too, so the row schema does not depend on
            # whether the lane happened to answer. MEASURED 2026-12-03: this key
            # was present on every up row and ABSENT on every down row, so a
            # consumer did `row["slots_cause"]` and got a KeyError on exactly the
            # lanes it most needed to explain.
            "slots_cause": None,
            "slot_ctx": None,
            "slot_ctx_slots": None,
            # Carried on the down row too, for the reason slot_ctx_slots is: a
            # consumer reading row["parked"] must not get a KeyError on exactly
            # the lanes it most needs to explain.
            "parked": None,
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
    # A lane that never answered is a LOCAL lane whose residency could not be
    # established -- not an absent lane. MEASURED 2026-12-03 at 297081a: with
    # the predicate below, a fleet of one live hosted lane plus two dead local
    # lanes produced no local-capacity line at all, because `local` came out
    # empty and the whole host block is nested in `if local:`. The fleet read
    # smaller and healthier than the truth, which is the expensive direction
    # for a report whose job is to keep work off billed lanes.
    # Hosted portals are excluded by KIND rather than by the all-None shape,
    # because a down lane wears that same shape.
    local = [r for r in rows
             if r.get("kind") not in HOSTED_KINDS
             and (r["resident"] is not None or r["resident_error"] is not None
                  or not r["reachable"])]
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
        # `serving` is load-bearing here, not decoration. MEASURED 2026-11-29:
        # a server answering /health 503 (still loading) also answers /slots
        # 200 with every slot free, so this line announced "every slot free"
        # for a server that could not yet take a single request -- the
        # overstatement in the same direction as the missing partial line, and
        # the reason both arms of this predicate are pinned.
        spare = [r for r in local
                 if r.get("slots") is not None and r.get("serving")
                 and r["slots"][0] > 0 and r["slots"][1] == 0]
        if spare:
            # A QUALIFIED certificate. MEASURED 2026-12-04 against the live
            # fleet: the one lane this line certified was `3090-defiant 0/4`,
            # while slot 3 held `id_task 158` with 19 prompt tokens ingested and
            # 0 processed, unchanged over nine reads across ~40 seconds. The
            # busy flags were true and the word "every slot free" was still an
            # overstatement, because a slot holding an assigned task is not
            # evidence of freedom. This is the same defect as the missing
            # partial line and the n_ctx scope note: a real number reported
            # without the fact that decides whether it can be used.
            #
            # The wording states the reservation and does NOT call the slot
            # busy. Which of the two it is -- reserved work about to prefill, or
            # state left behind by a finished task -- cannot be decided from the
            # /slots document, and picking either would be the same defect
            # facing the other way. The number and its qualifier travel
            # together; a lane with no parked slot renders exactly as before.
            def _park_note(row):
                parked = row.get("parked")
                if not parked:
                    return ""
                tasks = ", ".join(
                    "slot %s holds task %s" % (p.get("slot"), p.get("id_task"))
                    for p in parked)
                return " [%s -- not counted busy, not counted free]" % tasks

            print("[%d] loaded local server(s) with every slot free (MEASURED "
                  "spare capacity: %s)"
                  % (len(spare), ", ".join(
                      "%s 0/%d%s%s" % (r["lane"], r["slots"][0],
                                      _ctx_suffix(r), _park_note(r))
                      for r in spare)))
        # PARTIAL capacity is capacity too. MEASURED 2026-11-29: a loaded
        # llama.cpp server answering (4, 1) -- three slots free, one busy --
        # matched NO line in this report, because the predicate above demands
        # EVERY slot free. To a router the summary then reads identically to a
        # saturated server, which is precisely the fall-through to a billed
        # lane this tool exists to prevent. The model is resident either way,
        # so a partial server carries the same guarantee the all-free line is
        # built on: it cannot fail to load. Only lanes whose /slots was READ
        # appear here; a 401 lane is neither partial nor full.
        partial = [r for r in local
                   if r.get("slots") is not None and r.get("serving")
                   and r["slots"][0] - r["slots"][1] > 0 and r["slots"][1] > 0]
        if partial:
            print("[%d] loaded local server(s) with free slots (MEASURED "
                  "partial capacity: %s)"
                  % (len(partial), ", ".join(
                      "%s %d/%d free%s" % (r["lane"],
                                           r["slots"][0] - r["slots"][1],
                                           r["slots"][0],
                                           _ctx_suffix(r))
                      for r in partial)))
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
            # EVERY LANE'S OWN KNOB, and only the cause that was measured.
            # Two defects lived in the three lines this replaces:
            #
            # 1. It hinted `key_env_name(unnamed[0])` -- the FIRST unnamed lane
            #    only. MEASURED 2026-12-03 with q38 and 3090-inkling both 401:
            #    the line read "set FLEET_LANE_API_KEY_Q38 or FLEET_LANE_API_KEY"
            #    while the second lane needs FLEET_LANE_API_KEY_3090_INKLING,
            #    which appeared nowhere. The operator sets the one knob printed,
            #    reruns, and 3090-inkling is still dark with the identical
            #    sentence printed again -- the report asks for an action it
            #    cannot accept.
            # 2. It blamed the missing key for every failure. A timeout, a 500,
            #    a 404 and a wrong-shaped payload all got the same sentence; see
            #    the SLOTS_CAUSE_* block above for the six-arm measurement.
            #
            # Grouped by cause so a lane is never explained by a cause it did
            # not have. `auth` is the only bucket where a key is the fix.
            by_cause = {}
            for r in gated:
                c = r.get("slots_cause") or slots_cause(r.get("slots_error"))
                by_cause.setdefault(c, []).append(r)
            hints = []
            authrows = by_cause.pop(SLOTS_CAUSE_AUTH, [])
            unnamed = [r["lane"] for r in authrows if not r.get("key_source")]
            if unnamed:
                # The PHRASE "no API key configured" is load-bearing: two
                # sibling pins assert it, and it is TRUE for this bucket -- an
                # auth refusal with no key set really is an unconfigured key.
                # My first version replaced it and broke both, which is the
                # over-broad-guard mistake in reverse: the fix widened the
                # CAUSE analysis and needlessly changed wording that was right.
                # Every unnamed lane's own knob, then the shared fallback.
                knobs = ", ".join(key_env_name(l) for l in unnamed)
                hints.append("no API key configured; set %s (or one shared %s)"
                             % (knobs, LANE_KEY_ENV))
            for r in authrows:
                if r.get("key_source"):
                    hints.append("%s sent %s and the server refused it"
                                 % (r["lane"], r["key_source"]))
            # Everything else is NOT a key problem, and saying so is the point:
            # an operator must not be sent to set a variable that cannot help.
            for cause, rs in sorted(by_cause.items()):
                lanes = ", ".join(r["lane"] for r in rs)
                why = {
                    SLOTS_CAUSE_TIMEOUT: "the server did not answer /slots in time",
                    SLOTS_CAUSE_HTTP: "the server answered /slots with an HTTP error",
                    SLOTS_CAUSE_SHAPE: "the server answered /slots with a payload that is not a slot list",
                    SLOTS_CAUSE_UNREACHABLE: "the server could not be reached",
                    SLOTS_CAUSE_UNKNOWN: "the /slots read failed for an unrecognised reason",
                }.get(cause, "the /slots read failed")
                hints.append("%s: %s -- no API key fixes this (%s)"
                             % (lanes, why, rs[0].get("slots_error")))
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
