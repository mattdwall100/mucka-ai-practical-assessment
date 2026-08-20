"""Service layer — the hot path, top to bottom:
normalise/hash -> retrieval (few-shot) -> LLM -> safety floor -> store once.
"""
import hashlib
import json
import os
import re
import time

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# load .env if present — 5 lines beats a python-dotenv dependency
try:
    for _line in open(".env"):
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))
except FileNotFoundError:
    pass

PROMPT_VERSION = "v1"
LABELS = ("routine", "same_day", "emergency")
RANK = {l: i for i, l in enumerate(LABELS)}
SCORE_FLOOR = {"same_day": 40, "emergency": 80}  # score bands 0-39 / 40-79 / 80-100


# --- normalisation / idempotency key ------------------------------------------

def normalise(body: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", body.lower())).strip()


def body_hash(body: str) -> str:
    return hashlib.sha256(normalise(body).encode()).hexdigest()


# --- safety rules: deterministic, escalate-only -------------------------------
# An LLM can be talked out of an emergency by "no rush, but there's a smell of
# gas". These cannot. Applied AFTER the LLM; they only ever raise.

SAFETY_RULES = [
    # Transcripts and thumb-typing mangle exactly the words this rule depends on
    # ("smel gass", "gaz"). Letters may repeat or drop: a safety rule that only fires
    # on correct spelling is a rule that fails on the messages that matter most.
    ("gas_smell", "emergency", re.compile(
        r"sme+l+\w*.{0,25}\bga+(s+|z)\b|\bga+(s+|z)\b.{0,25}(sme+l+|leak|escap)", re.I)),
    ("carbon_monoxide", "emergency", re.compile(r"carbon monoxide|\bco (alarm|detector)", re.I)),
    ("flood_burst", "emergency", re.compile(
        r"flood|burst (pipe|tank|main)|(pouring|gushing) (out|through|down)|water .{0,20}through the ceiling", re.I)),
    ("fire_electric", "emergency", re.compile(r"sparks|burning smell|smell of burning", re.I)),
    ("vulnerable_occupant", "same_day", re.compile(
        r"\b(baby|newborn|toddler|elderly|disabled|oxygen|pregnant|wheelchair|\d{2}\s?(year|yr)s?[ -]old)\b"
        r"|\b(mum|mother|dad|father|gran|nan|grandma|grandad)\b.{0,30}\b(8\d|9\d|10\d)\b", re.I)),
]

# A vulnerable occupant plus total loss of heat is an emergency though neither half is
# one alone. Driven by real held-out misses — "Mum is 89 and there is no heating at all",
# "three week old baby in the house" — messages containing no alarm word whatsoever.
NO_SERVICE = re.compile(
    r"no heating|no hot water|heating (is )?(off|not working)|boiler.{0,20}locked? out"
    r"|house is freezing|no heat\b", re.I)


def apply_safety_floor(body: str, urgency: str):
    """Returns (possibly-raised urgency, names of rules that matched)."""
    floor, hits = urgency, []
    for name, level, rx in SAFETY_RULES:
        if rx.search(body):
            hits.append(name)
            if RANK[level] > RANK[floor]:
                floor = level
    if "vulnerable_occupant" in hits and NO_SERVICE.search(body):
        hits.append("vulnerable_no_service")
        floor = "emergency"
    return floor, hits


# --- retrieval: the corpus IS the adaptivity mechanism ------------------------

class Retriever:
    """TF-IDF char n-grams shrug off typos, ALL-CAPS and [inaudible] noise.
    # ponytail: full refit on every added label; fine to ~10k rows, incremental index after.
    """

    def __init__(self):
        self.rows, self.vec, self.mat = [], None, None

    def fit(self, rows):
        self.rows = rows
        if not rows:
            self.vec = self.mat = None
            return
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.mat = self.vec.fit_transform([r["body"] for r in rows])

    def retrieve(self, body, k=3, exclude_group=None):
        if self.vec is None:
            return []
        sims = linear_kernel(self.vec.transform([body]), self.mat)[0]
        out = []
        for i in sims.argsort()[::-1]:
            r = self.rows[i]
            if exclude_group is not None and r["dup_group"] == exclude_group:
                continue  # leakage guard: never retrieve yourself or your near-twin
            out.append({"body": r["body"], "urgency": r["urgency"], "sim": float(sims[i])})
            if len(out) == k:
                break
        return out


# --- LLM classification -------------------------------------------------------

SYSTEM_PROMPT = """You triage inbound customer messages (SMS / WhatsApp / transcribed voicemail) \
for the office manager of a UK plumbing & heating firm. Decide how urgently a human must see each message.

Urgency levels:
- emergency (score 80-100): danger to people or property RIGHT NOW — gas smell, carbon monoxide, \
active flood or burst pipe, water coming through a ceiling, burning smell or sparks.
- same_day (score 40-79): needs action today — total loss of heating or hot water, boiler locked out, \
a contained but active leak, a vulnerable occupant (baby, elderly, ill) affected by a fault.
- routine (score 0-39): everything else — quotes, bookings, billing, thanks, compliments, \
non-urgent repairs ("whenever you're passing").

Judge facts, not tone: customers downplay real problems — "not urgent, but there's a smell of gas" \
is still an emergency. A message may contain several intents; its urgency is the maximum over its parts. \
Voicemail transcripts contain transcription errors. Arrival time is given; out-of-hours arrival matters.

Some examples carry the office manager's own past label — stay consistent with her judgement.

Respond in JSON: urgency, intent (list of quote/booking/billing/complaint/job_update/other), \
score (integer within the band), reason (one short sentence)."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "triage",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "urgency": {"type": "string", "enum": list(LABELS)},
                "intent": {"type": "array", "items": {"type": "string", "enum": [
                    "quote", "booking", "billing", "complaint", "job_update", "other"]}},
                "score": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["urgency", "intent", "score", "reason"],
        },
    },
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()  # needs OPENAI_API_KEY
    return _client


def _model():
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def llm_classify(body, channel, received_at, shots):
    parts = [f"Example (labelled {s['urgency']} by the office manager):\n{s['body']}\n" for s in shots]
    parts.append(f"Message to classify:\nchannel: {channel}\nreceived: {received_at}\n{body}")
    for attempt in range(3):
        try:
            resp = _get_client().chat.completions.create(
                model=_model(),
                temperature=0,
                response_format=RESPONSE_FORMAT,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": "\n".join(parts)}],
            )
            out = json.loads(resp.choices[0].message.content)
            if out["urgency"] not in LABELS:
                raise ValueError(f"bad urgency {out['urgency']!r}")
            out["score"] = max(0, min(100, int(out["score"])))
            return out
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# --- orchestration ------------------------------------------------------------

def classify_message(repo, retriever, msg, exclude_group=None):
    """msg: {message_id, received_at (ISO), channel, from_name, body}. Idempotent on body hash."""
    h = body_hash(msg["body"])
    repo.upsert_message({**msg, "body_hash": h})
    cached = repo.get_classification(h)
    if cached:
        # Re-apply the safety rules to the cached LLM verdict rather than trusting the
        # stored one: rules are free and get edited often, LLM calls cost money and don't.
        # A rule fix therefore takes effect immediately, with no re-classification run.
        floored, hits = apply_safety_floor(msg["body"], cached["llm_urgency"])
        score = cached["score"]
        if floored != cached["llm_urgency"]:
            score = max(score, SCORE_FLOOR[floored])
        return {**cached, "urgency": floored, "score": score,
                "intent": json.loads(cached["intent"]), "rule_hits": hits,
                "message_id": msg["message_id"], "cached": True}

    shots = retriever.retrieve(msg["body"], k=3, exclude_group=exclude_group)
    out = llm_classify(msg["body"], msg["channel"], msg["received_at"], shots)
    llm_urgency = out["urgency"]
    floored, hits = apply_safety_floor(msg["body"], llm_urgency)
    if floored != llm_urgency:
        out["urgency"] = floored
        out["score"] = max(out["score"], SCORE_FLOOR[floored])
        out["reason"] += f" [safety rule: {', '.join(hits)}]"

    repo.insert_classification({
        "body_hash": h, "urgency": out["urgency"], "intent": json.dumps(out["intent"]),
        "score": out["score"], "reason": out["reason"], "rule_hits": json.dumps(hits),
        "llm_urgency": llm_urgency, "model": _model(), "prompt_version": PROMPT_VERSION,
    })
    return {**out, "body_hash": h, "rule_hits": hits, "llm_urgency": llm_urgency,
            "message_id": msg["message_id"], "cached": False}


def add_label(repo, retriever, message_id, urgency):
    """Override capture: the manager's correction becomes a retrieval example. Returns corpus size."""
    m = repo.get_message(message_id)
    if m is None:
        raise KeyError(message_id)
    repo.add_corpus_row({
        "message_id": message_id, "body": m["body"], "body_hash": body_hash(m["body"]),
        "urgency": urgency, "dup_group": repo.next_dup_group(), "source": "override",
    })
    retriever.fit(repo.corpus_rows())
    return repo.corpus_size()
