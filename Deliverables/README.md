# Triage: how urgently does a human need to see this message?

Inbound SMS / WhatsApp / voicemail for a UK plumbing firm, scored on arrival. This README is the
decision log; `evaluations.md` is the evaluation write-up; `memo.md` is the founder-facing
version; `NOTES.md` is the raw timeline.

## Run it

```
pip install -r requirements.txt          # Python 3.14

python ml/preprocess.py     # both CSVs (in artifacts/) -> data/processed/
python ml/cold.py           # train + calibrate -> data/config/v1/
python ml/predict.py        # -> predictions.csv at the repo root
python evals/evaluate.py    # the scoreboard   (set EVAL_SKIP_HOT=1 first, see below)
python evals/traps.py       # 22 hand-written adversarial cases
```

Those five run from a clean checkout in that order, offline, on stock Python 3.14 — no API key, no
manual steps (`data/` is gitignored, so preprocess goes first; the input CSVs are committed under
`artifacts/`). Skip the LLM channel in the scoreboard by setting `EVAL_SKIP_HOT=1`:

```
$env:EVAL_SKIP_HOT = "1"          # PowerShell
export EVAL_SKIP_HOT=1            # bash / zsh
```

`ml/predict.py` writes a **cold-path** `predictions.csv` to the repo root. The shipped
`Deliverables/predictions.csv` is the **hot-path** one (LLM reasons per row) and is only reproducible
with a key, via `backfill.py` below — expect the two to differ.

The LLM path is optional and needs a key (`export OPENAI_API_KEY=...` on bash):

```
$env:OPENAI_API_KEY = "sk-..."     # OPENAI_MODEL defaults to gpt-4o-mini
python backfill.py                 # LLM classify holdout -> predictions.csv (~300 calls, resumable)
python evals/evaluate.py           # same scoreboard, with the hot channel included
uvicorn app.api:app --port 8000    # POST /messages, GET /inbox, POST /messages/{id}/label, POST /retrain
```

Layout: `app/` hot path (SQLite + FastAPI + LLM), `ml/` cold path (preprocess, train, offline
predict), `evals/evaluate.py` the scoreboard. Both evals are deliverables and answer different
questions (table below): root `evaluate.py` scores the shipped hot channel on all 300 labelled rows
and is the source of the memo's headline numbers; `evals/evaluate.py` compares hot against cold on
the held-back half.

---

## The decisions

**Sorter and safety net, never a filter.** Nothing is hidden; the system changes the *order* she
reads in and raises a banner on the few that cannot wait. She already reads everything — the value
on offer is ordering, not suppression. Cost: near zero. Benefit: our worst failure downgrades from
"missed forever" to "seen twenty minutes late", which is what makes this shippable on this much
evidence.

**Keep the 3 given labels; put intent on a second axis.** `routine|same_day|emergency` are inferred
from the labelled rows, not given in the brief, and `emergency` genuinely mixes life-safety with
property damage. Invent a 4th level and the eval becomes unmeasurable — 300 labels are the only
ground truth there is. So a hot quote is `intent=quote AND urgency=same_day`, a filter rather than a
class. Intent is reported **unevaluated**; there are no labels for it.

**Safety rules escalate only.** A deterministic floor on gas / CO / flood / burning sits above the
model. Two reasons, and the second is the real one: (a) 91 messages contain de-escalating language —
an LLM can be talked out of an emergency by *"no rush, but there's a smell of gas"*; (b) **the
severest hazard families have zero labels.** CO-alarm-and-evacuated appears once in `messages`,
unlabelled, and twice in the holdout. Indoor gas smell: 5, unlabelled. A supervised model provably
cannot learn these from this data. The rule layer is not a shortcut, it is the only thing covering
them.

**Unlabelled is not routine.** Labelling is a random 24% sample, so the 952 unlabelled rows contain
silent gas leaks. Training on them as negatives teaches the model that gas leaks are fine.

**Labels are one busy person's Tuesday, not truth.** 9 of 225 near-identical labelled pairs disagree
with each other. That caps achievable accuracy on near-identical text and makes *consistency* the
prize rather than correctness. Also: "boiler hissing and smells hot" is labelled `same_day` all
three times, which we think is simply a wrong label. Every number below is agreement with a noisy
judge and is quoted that way.

**Group by template before believing any score.** 174 of 300 holdout messages have a >0.7-Jaccard
twin in `messages`; bodies are a template core plus spliced noise. The gold split is 50/50
`StratifiedGroupKFold` over near-duplicate clusters, and retrieval never returns a message's own
cluster. `1NN-cosine` stays in the scoreboard as the **leakage exhibit** — it scores well by
memorising templates, which is the point of showing it.

**Combine the two CSVs for diagnostics, never for fitting.** They are one dataset: IDs
m00001–m01552, no gaps, matching channel mix and out-of-hours rate. So the holdout is IID, *not* a
future period — a random-split CV will predict the holdout score well while overstating production,
where tomorrow's message is new. Fitting on holdout text would be legitimate transductively and
would buy nothing (we already sit on 952 unlabelled rows), so we do not.

**Operating point set by the alert budget, not the ROC curve.** If 30% of the inbox is flagged,
nothing is urgent by week three. ~10 flags/day out of 60–120, and recall is reported *at* that
budget.

**Absolute scores, so reordering is an `ORDER BY`.** Score once at arrival; the inbox is
`ORDER BY score DESC, received_at ASC`. Comparative ranking would invalidate the order on every
arrival and create a re-ranking job that does not need to exist. Cost ≈ £20–50/year per business;
it influenced nothing here.

**Adaptivity is retrieval, not retraining.** Each message is queried (TF-IDF char n-grams, so typos
and `[inaudible]` still match) against the labelled corpus; the 3 nearest go into the prompt with
the manager's own labels. Every override becomes a corpus row, so the system improves with no
training task — day one it is zero-shot, six months in most messages arrive with a neighbour she has
already ruled on. The pool is de-conflicted first: a contradiction in the few-shot block is worse
than no example.

**Two paths, and the cold one never touches live routing.** Hot: rules → retrieval → LLM →
escalate-only floor → stored once, idempotent on normalised-body hash (the near-duplicate density
that ruins the evaluation is a straight win here). Cold: retrain TF-IDF+LR, recalibrate the
threshold, emit a versioned artifact under `data/config/vN/`, promote by writing `current.txt`, roll
back by editing it.

**Where we sit on the personalisation ladder:** (1) prompt+rules → (2) recalibrate threshold →
(3) few-shot on their own labels → (4) small supervised model → (5) fine-tuning. We built 1–4 and
stop there. Rung 4 is not a cheaper rung 3; it is a **second opinion with a different failure
mode** — the LLM knows "carbon monoxide" from world knowledge and cannot learn her habits; the model
learns her habits and cannot know CO. Rows where they disagree are the highest-information labels
available and belong in a review queue. No per-customer fine-tuning at 30k messages/year.

**Things the brief did not mention that change the answer.** Urgency ≠ actionability: 26% of traffic
is out-of-hours, and a gas smell at 22:40 needs an on-call phone and the National Gas Emergency
number (0800 111 999), not a higher score — the firm is not even the right first responder.
`from_name` is 40 first names × 20 surnames recombined, so there is no repeat-customer signal to
extract and building one would be fitting noise — worth saying out loud, because in the real system
that history is probably the strongest feature available. Bodies carry PII (addresses, key-safe
locations) and the hot path sends it to a hosted API. Many messages are multi-intent; single-label
classification loses that.

---

## The numbers

**There are two evaluations, and they answer different questions.** Read this before comparing
their emergency counts, because they differ legitimately.

| | `evaluate.py` (root) | `evals/evaluate.py` |
|---|---|---|
| asks | how good is the shipped hot channel? | is the LLM worth its cost over a cheap model? |
| scored on | all 300 labelled rows, leave-cluster-out retrieval | 150 held-back rows (`gold_test`) |
| emergencies | n=11 | n=6 |
| why not one eval | the LLM never trains, so all 300 is fair to it | the cold model does train, so it needs a held-out half |

`memo.md` quotes the first (n=11 is the better sample, and it is the system that actually ships).
The scoreboard below is the second, because comparing hot against cold requires the split.

150 held-back gold rows (122 routine / 22 same_day / 6 emergency), grouped split, no near-twin
spanning train and test. `EVAL_SKIP_HOT=1 python evals/evaluate.py`:

```
always-routine  acc 81%  emer 0/6 (CI 0-39%)   same 0/22 (CI 0-15%)   false-alarms 0/122
rules-only      acc 81%  emer 6/6 (dagger)     same 0/22 (CI 0-15%)   false-alarms 6/122
1NN-cosine      acc 83%  emer 0/6              same 11/22             false-alarms 8/122  <- leakage exhibit
cold TFIDF+LR   acc 86%  emer 0/6              same 9/22              false-alarms 2/122
cold+rules      acc 86%  emer 6/6 (dagger)     same 9/22              false-alarms 8/122
hot LLM+rules   acc 90%  emer 6/6 (dagger)     same 16/22 (CI 52-87%) false-alarms 9/122
```

Adversarial trap set, 22 hand-written messages held out from everything (`python evals/traps.py`):

```
cold  exact 16/22   under-ranked 4   over-ranked 2
hot   exact 20/22   under-ranked 0   over-ranked 2
```

**(dagger) The emergency numbers are contaminated.** The `vulnerable_no_service` rule (vulnerable
occupant + total loss of heat -> emergency) was written *after reading the four emergency misses in
gold_test*. That is fitting to the test set. It is the reason every 6/6 above reads 6/6:
rules-only, cold+rules and hot all clear the class on the strength of that one rule. The honest
out-of-sample figure before the rule existed was **2/6**; afterwards it reads 6/6, and 6/6 is no
longer a clean estimate. The rule is defensible on its own terms -- a three-week-old baby or an 89-year-old with no heat genuinely is an emergency,
and saying so needs no training data -- but the number it produces is not evidence. Only a fresh
labelled sample fixes it.

Read the top two rows first. Always-`routine` scores **81% accuracy with 0% emergency recall**,
which is why accuracy is the wrong headline. And the cold model alone gets **0/6 emergency** — with
~7 emergency training rows it cannot learn the class. That is the whole argument for the rule layer
and the LLM, stated as a measurement rather than an opinion.

Those are argmax numbers. **At the shipped operating point** (`p_urgent >= 0.66` or a rule fires =
11.4 flags/day at 90 messages/day) the same artifact flags **6/6 emergencies (dagger), 6/22 same_day,
12/28 urgent, at 7 false flags out of 122 routine.** Note what that means: all six emergencies sit
*below* the 0.66 model threshold — every one is caught by a rule, four of them by the contaminated
one. The model contributes nothing to emergency recall at the operating point.

`memo.md` quotes the all-300 argmax figures from root `evaluate.py`, not this row — see the two-eval
table above.
*Known gap: the scoreboard prints the argmax row; the at-budget row above was computed by hand
against the shipped artifact, because `ml/cold.py` calibrates on training CV (21/22 urgent, 5/5
emergency in `config.json`) and never prints the gold_test figure. One row of code, and until it
exists the operating-point numbers drift out of date with the rules — which is exactly what had
happened to them.*

The downgrade list — every urgent message ranked too low, printed in full — is the output that
matters most, and the evaluation prints it. The worst one: *"Mum is 89 and there is no heating at
all, the house is freezing and she's not well."* No alarm word anywhere, so no keyword rule will
ever catch it; reading meaning is exactly the LLM's job.

**Where the LLM earns its cost: `same_day`, not emergencies.** 16/22 against the cold path's 9/22.
Emergencies are caught by deterministic rules that cost nothing and cannot be talked out of firing.
That split is the design: rules for the catastrophic-and-rare, the LLM for the ambiguous-and-common.
On the trap set the hot channel under-ranks *nothing* -- its two errors both over-rank (a gas leak
resolved last month, and an oxygen user whose boiler locked out), which is the stated preference.

Hot and cold disagree on **25/150 rows** (~15 messages/day at 90/day), almost all of them same-day
work the cold model missed. Routing disagreements to a human is therefore a cheap real safety net.

**n=6.** Every emergency interval spans tens of points. The CI does the arguing, not the point
estimate.

## Status

At the hard stop the OpenAI account had hit a request cap, so `predictions.csv` shipped from the
cold path and the LLM numbers were blank. **The key was replaced afterwards and both were filled in**
— the resumability claim was then tested for real: every previously scored body came back as a
SQLite cache hit and only the remainder was re-called.

`predictions.csv` (300 rows) now comes from the hot path via `backfill.py`, carrying the LLM's own
one-line reason per row: 229 routine / 61 same_day / 10 emergency.

Two things the resumed run changed in the code, both worth knowing:

- **Safety rules now re-apply to cached LLM verdicts** rather than the stored floored one
  (`app/service.py`). Rules are free and get edited often; LLM calls cost money and don't. A rule
  fix now takes effect with no re-classification run — which is what made the post-fix evaluation a
  zero-cost re-run rather than another 150 calls.
- **The safety regexes were typo-brittle** and the trap set caught it: `smell (of )?gas` never fires
  on *"smel gass"*. They now tolerate repeated and dropped letters (`sme+l+`, `ga+(s+|z)`). Checked
  against false positives — "gas hob", "Gas Safe registered", "gas safety certificate" all stay
  routine.

## With more time

Ranked by what would actually change the answer.

1. **Get an uncontaminated emergency number.** The 6/6 above was measured after a rule written to
   fix the misses it is scored on, and n=6 was never enough anyway. Label 50 more emergencies —
   worth more than any modelling change — and prefer the messages the system is *least* confident
   about over random ones. The override endpoint is how that happens for free in production.
2. **Fix `same_day` recall.** The real weak spot: 16/22 hot, 9/22 cold. Loss-of-service ("shower has
   gone completely cold") and contained leaks ("slow drip, filled a bucket overnight") are the
   recurring pattern, and they are the messages that make a customer feel ignored. The trap set
   isolates both.
3. **Redact PII before the API call.** Addresses, phone numbers and key-safe locations go to OpenAI
   in full today. Cheap to strip, and it removes a real objection rather than documenting one.
4. **Run it silently for 2–4 weeks** and count what it *would* have buried. No customer affected.
5. **A time-based split alongside the grouped one.** The holdout is IID, so grouped CV predicts the
   holdout score but overstates production. The gap between the two splits is the honest finding,
   and it is not measured yet.
6. **Group the cold path's CV by template.** `ml/cold.py` de-dupes on exact normalised core, but 813
   near-duplicate pairs survive into its 5-fold CV — so the calibration numbers in `config.json` are
   optimistic. Same `GroupKFold` the preprocessing already uses.
7. **Split promote from retrain.** `POST /retrain` currently promotes in the same call. A
   metric-gated auto-promote can silently degrade safety behaviour on a bad night of labels; a human
   clicking promote is cheap.
8. **The ML second opinion inside the hot path** (`final = max(rules, llm, ml)`, disagreement stored
   and surfaced as the review queue). Designed, and both models exist — but they are only compared
   offline in `evals/evaluate.py`, not fused at request time.
9. **Out-of-hours routing**, and **age-decay in the sort** (`effective = score + decay(age, class)`,
   which fixes starvation of old routine messages with arithmetic rather than a model call).
10. **Repeat-contact signal** ("third time I've chased") — needs the sender identity this data
   withholds.
11. **Scale seams**, all marked `# ponytail:` in code: O(n²) Jaccard dedup and full TF-IDF refit per
    label (fine to ~10k rows, MinHash beyond), sequential LLM calls in backfill (thread pool),
    SQLite (the seam is `connect()` + one `Repo`), and the cold path reading a CSV where production
    would read the corpus table.

## Known inconsistencies, stated rather than hidden

- Hot and cold have **two different `normalise` functions and two different de-duplication
  routines** (`app/service.py` vs `ml/common.py`), and they resolve label conflicts differently —
  hot drops contradictory clusters, cold escalates them. They should be one routine with two
  consumers.
- The duplicate rule lists in `ml/common.py` have been deleted — `app/service.py` is now the single
  copy both channels import. The de-escalation trap is still handled only by the LLM prompt, so the
  offline fallback has no defence against it; the trap set confirms both channels pass the four
  de-escalation cases anyway, because the safety rules fire regardless of tone.
- `ml/preprocess.py` pseudo-labels 438 unlabelled rows by cosine kNN into the training set — 76% of
  `train.csv` — using the same nearest-neighbour mechanism the scoreboard calls the leakage exhibit.
  It abstains below 0.6 similarity and mints almost no emergencies, but it deserves a harder look
  than the clock allowed.
