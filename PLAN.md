# Project plan

Unifies the design decisions already in [README.md](README.md) with the two-model hot/cold
architecture. This document is the build order and the scope contract. Hard stop **15:30**.

---

## 1. The system in one paragraph

Every inbound message is scored once, on arrival, by a **hot path** that combines three independent
estimators: deterministic safety rules, an LLM call carrying retrieved few-shot examples, and a
small supervised model trained on the business's own accumulated labels. Nothing is ever hidden —
the score drives an `ORDER BY`, and the rules raise a banner on the few that cannot wait. A
**cold path**, triggered by an API call and never touching live routing, retrains the supervised
model from the database, re-calibrates the threshold against the alert budget, evaluates the
candidate against the incumbent *and* against the LLM, and emits a versioned config that a human
promotes deliberately.

## 2. Why two models instead of one

They fail differently, and that is the entire point.

| | LLM + retrieved few-shot | Supervised model (TF-IDF → LR) |
|---|---|---|
| Cold start | Works day one, zero labels | Needs ~200+ labels before it means anything |
| Knows "carbon monoxide" is bad | Yes, from world knowledge | Only if it appears in training data — and **it never does**, that hazard family has zero labels |
| Learns *this manager's* judgement | Only via examples in the prompt | Directly, from every override |
| Latency / cost | ~1s, hosted, per-message cost | ~1ms, in-process, free |
| PII exposure | Body leaves the building | Never leaves |
| Deterministic | No | Yes |
| Auditable | Reason string, post-hoc | Coefficients *are* the reason |

The supervised model is not a cheaper replacement for the LLM. It is a **second opinion from a
different failure mode**, and the cases where the two disagree are the most valuable messages in
the system — see §5.

## 3. Hot path

Per message, on arrival, target < 2s.

```
body
 ├─ normalise() ──────────────► hash ──► cache hit? ──► return stored classification
 ├─ 1. safety rules            deterministic, escalate-only floor
 ├─ 2. retrieval               TF-IDF char n-grams, 3 nearest from de-conflicted corpus
 ├─ 3. LLM classify            rubric + few-shot ──► {urgency, intent, score, reason}
 ├─ 4. ML second opinion       promoted model ──► {urgency, proba}   (None if no model yet)
 └─ 5. combine ──────────────► {urgency, intent, score, reason, route, agreement}
                                stored once, idempotent on body hash
```

### The combine step

```
ordinal = {routine: 0, same_day: 1, emergency: 2}

final    = max(rules_floor, llm_urgency, ml_urgency)      # escalate-only, all three
agreement = "agree" if llm_urgency == ml_urgency else "split"
score     = llm_score if agreement == "agree" else max(llm_score, ml_proba_urgent)
```

Three properties fall out of this, and each maps to a stated commitment:

- **Escalate-only across all three estimators.** Consistent with README §"Safety rules escalate
  only" and with the stated preference for false positives over false negatives. No component can
  talk another one down.
- **Graceful cold start.** With no promoted model, `ml_urgency` is `None` and `max()` degrades to
  rules + LLM — exactly the day-one system the README already describes. The supervised model is
  additive, never load-bearing.
- **Disagreement is captured, not discarded.** `agreement="split"` is stored on the row.

`route` (`call_now` / `first_thing` / `queue`) is derived from `received_at` + `final` by lookup —
the urgency-is-not-actionability point from README §"Things the brief didn't mention". No model
involved.

## 4. Cold path

Offline, no latency budget, exposed as an API so it can be run on demand or by cron.

```
POST /admin/retrain
 1. pull every row with a confirmed label      original labels + confirmations + overrides
 2. de-conflict                                group by normalised core, drop contradictory groups
 3. split                                      GroupKFold on normalised core  +  time-based split
 4. train                                      TF-IDF (word 1-2 + char_wb 3-5) → LogisticRegression
 5. evaluate                                   candidate vs incumbent vs LLM-historical (§6)
 6. recalibrate threshold                      to the alert budget, not to the ROC curve
 7. refresh few-shot pool                      the de-conflicted corpus from step 2 — same artifact
 8. emit versioned config                      status = candidate.  NEVER auto-promoted.

POST /admin/promote/{version}                  deliberate, diffable, reversible
```

Step 2 is shared with the hot path's retrieval pool — one de-confliction routine, two consumers.
A contradiction in the few-shot block is worse than no example, and a contradiction in the training
set is worse than a missing row; the same 9-of-225 disagreeing pairs poison both.

Step 3 is not optional. 174 of 300 holdout messages have a >0.7-Jaccard twin in `messages`. A
random split would report a memorisation score, and the retrain endpoint would happily promote a
model on the strength of it. **Grouping by normalised core is a correctness requirement of the
retrain API, not an evaluation nicety.**

The config artifact:

```json
{ "version": "...", "trained_at": "...", "corpus_rows": 0, "threshold": 0.0,
  "metrics": { "...": "per-split, per-class, with CIs" }, "status": "candidate|promoted|rolled_back" }
```

Promotion is a separate call because an automatic promote gated on a metric is a system that can
silently degrade its own safety behaviour on a bad night of labels. A human clicking promote is
cheap; a regression in emergency recall is not.

## 5. The closed loop — why disagreement is the product

Every message where `agreement == "split"` is surfaced in the manager's review queue. She rules on
it in one tap. That override is:

- a **free label** from the person whose judgement *is* the ground truth (README already makes this
  point — this is the mechanism that acts on it),
- the **highest-information label available**, because it sits exactly where two independent
  estimators diverge,
- a new corpus row, so it improves the hot path's retrieval *immediately*, before any retraining.

This is active learning without the machinery. The corpus stops being frozen at 11 emergencies, and
the labelling effort is spent where it is worth the most instead of uniformly.

## 6. Evaluation

Same labelled rows, same splits, every system side by side. The headline comparison the brief and
the new architecture both demand is **ML vs hot path**.

| System | What it establishes |
|---|---|
| always `routine` | The floor. 82% accuracy, 0% emergency recall — proves accuracy is the wrong metric |
| rules only | How much of the value is free and deterministic |
| 1-NN cosine | **Leakage exhibit.** Will score suspiciously well by memorising templates |
| **ML only** | The cold-path model alone — the rung-4 estimator |
| LLM zero-shot | The rubric with no examples |
| LLM + few-shot | Adaptivity's contribution, isolated |
| **hot path (rules ∨ LLM ∨ ML)** | What actually ships |

Reported for each, on **both** the grouped split and the time split — the gap between them is
itself a finding:

- **Emergency and same_day recall with Wilson intervals.** `10/11 = 91% (62–98%)`. With n=11 the
  interval does the arguing; a bare point estimate would be dishonest.
- **Escalations per day at the chosen threshold**, so recall is quoted at a budget somebody could
  actually live with (5–10/day out of 60–120), not at the ROC optimum.
- **The downgrade list** — every urgent message called routine, printed with its full text. This is
  the page the founder reads.
- **LLM × ML agreement matrix** — how often the two independent estimators agree, and what the
  disagreements look like. Feeds §5 and answers "how much is the second model buying us".
- **Trap set** (`tests/traps.csv`, ~20 hand-written adversarial rows): de-escalating language plus
  a real hazard, a gas smell buried inside a compliment, `[inaudible]` truncation, `boier`/`teh`
  typos, ALL-CAPS, multi-intent. Scored and reported separately, unflatteringly.
- **The label-noise ceiling, stated first**, so every number above is read against it.

`intent` is reported as **unevaluated**. There are no labels for it and we will not pretend
otherwise.

## 7. Files

```
app/db.py         SQLite + repository.  The Postgres seam.
app/rules.py      safety floor + de-escalation phrases.  Pure regex, no deps.
app/corpus.py     normalise, de-conflict, TF-IDF retrieval.  Shared by hot and cold.
app/model.py      train / promote / load / predict.  Versioned artifacts.
app/service.py    the hot path: rules → retrieval → LLM → ML → combine.
app/api.py        FastAPI, thin.  POST /messages, GET /inbox,
                  POST /messages/{id}/label, POST /admin/retrain, POST /admin/promote/{v}
backfill.py       seed corpus from messages.csv, classify holdout → predictions.csv
evaluate.py       §6, top to bottom, prints its numbers
tests/traps.csv   the adversarial slice
memo.md           one page to the founder
```

`api.py` is nearly free: `backfill.py` and `evaluate.py` need `service` and `model` to exist
anyway, so the endpoints are thin wrappers over functions that already have to work. The retrain
endpoint is five lines calling `model.retrain(db)`.

## 8. Scope for today, and what gets cut first

Four graded deliverables: `README.md`, `predictions.csv`, the evaluation, `memo.md`. Everything
else is architecture, and architecture that costs a deliverable is a bad trade.

Build order — each step leaves the repo in a handable state:

1. `db.py`, `corpus.py`, `rules.py` — no LLM, no model, testable immediately
2. `model.py` + `evaluate.py` with **ML only** → real numbers on the board, no network, no key
3. `service.py` + LLM → hot path complete
4. `backfill.py` → `predictions.csv`
5. `api.py` → the retrain/promote surface
6. `memo.md`, README reconciliation, "with more time"

**Cut order if time runs short:** 5 before 3. The API surface is describable in the README without
being built; `predictions.csv` is not.

### Risks

- **300 LLM calls in `backfill.py`** is the main schedule risk. Mitigated by the normalised-hash
  cache — the near-duplicate density that ruins the evaluation is a straight win here — plus
  concurrency and resumability. If the key is unavailable or calls are slow, step 2 has already put
  a complete rules + ML system on disk and `predictions.csv` still ships.
- **PII leaves the building** on the hot path. Named as a deliberate decision in `memo.md`, not
  buried. The ML path never sends anything.

## 9. README reconciliation

Two edits the new architecture forces:

- **Line 119, "We stop at 3, and expect to stay there."** Now: we build rung 4, but as a *shadow
  second opinion inside the hot path*, not as the router. The justification changes from "few-shot
  captures the preference signal" to "an independent estimator with a different failure mode is
  worth more than a marginally better single one, and its disagreements are the labelling queue."
  We still do not climb to 5 — no per-customer fine-tuning.
- **README §"Two paths"** — the cold path description gains model retraining alongside threshold
  recalibration and few-shot refresh, and gains the promote/rollback endpoint.
