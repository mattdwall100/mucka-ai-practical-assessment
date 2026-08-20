# Triage: how urgently does a human need to see this message?

Inbound SMS / WhatsApp / voicemail for a UK plumbing firm, scored on arrival. This README is the
decision log; `memo.md` is the founder-facing version; `NOTES.md` is the raw timeline.

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
python evals/evaluate.py           # same scoreboard, hot channel included
uvicorn app.api:app --port 8000    # POST /messages, GET /inbox, POST /messages/{id}/label, POST /retrain
```

Layout: `app/` hot path (SQLite + FastAPI + LLM), `ml/` cold path (preprocess, train, offline
predict), `evals/evaluate.py` the scoreboard. Root `evaluate.py` is an earlier hot-only eval, kept
only because it is resumable.

## The design

**Sorter and safety net, never a filter.** Nothing is hidden; the system changes the *order* she
reads in and banners the few that cannot wait. Our worst failure downgrades from "missed forever" to
"seen twenty minutes late" — which is what makes this shippable on this little evidence.

**Two channels.** Hot: rules → retrieval → LLM → escalate-only safety floor → stored once,
idempotent on normalised-body hash. Cold: TF-IDF+LR retrained offline, recalibrated, emitted as a
versioned artifact under `data/config/vN/`; promote by writing `current.txt`, roll back by editing
it. The cold path never touches live routing.

**Safety rules escalate only, and they are not a shortcut.** 91 messages contain de-escalating
language, so an LLM can be talked out of an emergency by *"no rush, but there's a smell of gas"*.
More decisively, **the severest hazard families have zero labels**: CO-alarm-and-evacuated appears
once in `messages` unlabelled and twice in the holdout; indoor gas smell 5 times, unlabelled. A
supervised model provably cannot learn these here. A deterministic floor on gas / CO / flood /
burning is the only thing covering them.

**Adaptivity is retrieval, not retraining.** Each message is queried (TF-IDF char n-grams, so typos
and `[inaudible]` still match) against the labelled corpus; the 3 nearest go into the prompt with
the manager's own labels, de-conflicted first. Every override becomes a corpus row — day one it is
zero-shot, six months in most messages arrive with a neighbour she has already ruled on.

**Rules + LLM + model is three failure modes, not redundancy.** The LLM knows "carbon monoxide" from
world knowledge and cannot learn her habits; the cold model learns her habits and cannot know CO
(0/6 emergency recall, below). Rows where they disagree are the highest-information labels
available. On the personalisation ladder — prompt+rules → recalibrate → few-shot on their own labels
→ small supervised model → fine-tuning — we built the first four and stop.

**Operating point set by the alert budget, not the ROC curve.** If 30% of the inbox is flagged,
nothing is urgent by week three. ~10 flags/day out of 60–120, and recall is reported *at* that
budget. Scores are absolute, so the inbox is `ORDER BY score DESC, received_at ASC` — comparative
ranking would invalidate the order on every arrival. Cost ≈ £20–50/year per business; it influenced
nothing here.

## What the data actually is

**Labels are one busy person's Tuesday, not truth.** 9 of 225 near-identical labelled pairs disagree
with each other — a ceiling on achievable accuracy, and it makes *consistency* the prize rather than
correctness. "Boiler hissing and smells hot" is labelled `same_day` all three times, which we think
is simply wrong. Every number below is agreement with a noisy judge.

**Unlabelled is not routine.** Labelling is a random 24% sample, so the 952 unlabelled rows contain
silent gas leaks. Training on them as negatives teaches the model that gas leaks are fine.

**Group by template before believing any score.** 174 of 300 holdout messages have a >0.7-Jaccard
twin in `messages`; bodies are a template core plus spliced noise. The gold split is 50/50
`StratifiedGroupKFold` over near-duplicate clusters, and retrieval never returns a message's own
cluster. `1NN-cosine` stays in the scoreboard as the **leakage exhibit** — it scores well by
memorising templates, which is the point of showing it.

**The two CSVs are one dataset** (IDs m00001–m01552, no gaps, matching channel mix and out-of-hours
rate), so the holdout is IID, *not* a future period. Random-split CV predicts the holdout score
while overstating production, where tomorrow's message is new. Combined for diagnostics, never for
fitting.

**Kept the 3 given labels; put intent on a second axis.** `routine|same_day|emergency` are inferred
from the labelled rows, not given. `emergency` genuinely mixes life-safety with property damage, but
inventing a 4th level makes the eval unmeasurable — 300 labels are the only ground truth there is.
So a hot quote is `intent=quote AND urgency=same_day`, a filter rather than a class. Intent ships
**unevaluated**; there are no labels for it.

**Things the brief did not mention that change the answer.** Urgency ≠ actionability: 26% of traffic
is out-of-hours, and a gas smell at 22:40 needs an on-call phone and the National Gas Emergency
number (0800 111 999) — the firm is not even the right first responder. `from_name` is 40 first
names × 20 surnames recombined, so there is no repeat-customer signal and building one would be
fitting noise; in a real system that history is probably the strongest feature available. Bodies
carry PII (addresses, key-safe locations) and the hot path sends it to a hosted API. Many messages
are multi-intent; single-label classification loses that.

## The numbers

150 held-back gold rows (122 routine / 22 same_day / 6 emergency), grouped split, no near-twin
spanning train and test. `EVAL_SKIP_HOT=1 python evals/evaluate.py`:

```
always-routine  acc 81%  emer 0/6 (CI 0-39%)   same 0/22 (CI 0-15%)   false-alarms 0/122
rules-only      acc 79%  emer 2/6 (CI 10-70%)  same 0/22 (CI 0-15%)   false-alarms 6/122
1NN-cosine      acc 83%  emer 0/6              same 11/22             false-alarms 8/122  <- leakage exhibit
cold TFIDF+LR   acc 86%  emer 0/6              same 9/22              false-alarms 2/122
cold+rules      acc 86%  emer 6/6 (dagger)     same 9/22              false-alarms 8/122
hot LLM+rules   acc 90%  emer 6/6 (dagger)     same 16/22 (CI 52-87%) false-alarms 9/122
```

Read the top two rows first. Always-`routine` scores **81% accuracy with 0% emergency recall**,
which is why accuracy is the wrong headline. The cold model alone gets **0/6** — with ~7 emergency
training rows it cannot learn the class. That is the argument for the rule layer and the LLM, as a
measurement rather than an opinion.

**(dagger) The emergency numbers are contaminated.** The `vulnerable_no_service` rule (vulnerable
occupant + total loss of heat → emergency) was written *after reading the four emergency misses in
gold_test*. The honest out-of-sample figure before it existed was **2/6**. The rule is defensible on
its own terms — a three-week-old baby with no heat is an emergency, and saying so needs no training
data — but the 6/6 it produces is not evidence. Only a fresh labelled sample fixes that.

Those are argmax numbers. **At the shipped operating point** (`p_urgent >= 0.66` or a rule fires =
10.8 flags/day at 90/day) the same artifact flags **5/6 emergencies, 6/22 same_day, 11/28 urgent, at
7 false flags out of 122 routine.** Those are the figures `memo.md` quotes.
*Known gap: that row is computed inside `ml/cold.py`'s calibration but not printed beside the
scoreboard. One line of code — and it is the line the memo depends on.*

Adversarial trap set, 22 hand-written messages held out from everything (`python evals/traps.py`):

```
cold  exact 16/22   under-ranked 4   over-ranked 2
hot   exact 20/22   under-ranked 0   over-ranked 2
```

**The LLM earns its cost on `same_day`, not emergencies** — 16/22 against the cold path's 9/22.
Emergencies are caught by rules that cost nothing and cannot be talked out of firing. That split is
the design: rules for the catastrophic-and-rare, the LLM for the ambiguous-and-common. On the traps
the hot channel under-ranks nothing; both its errors over-rank, which is the stated preference.

The downgrade list — every urgent message ranked too low, printed in full with its text — is the
output that matters most to this audience, and the evaluation prints it. The worst: *"Mum is 89 and
there is no heating at all, the house is freezing and she's not well."* No alarm word anywhere, so
no keyword rule will ever catch it.

Hot and cold disagree on **25/150 rows** (~15/day at 90/day), almost all same-day work the cold
model missed — so routing disagreements to a human is a cheap real safety net.

**n=6.** Every emergency interval spans tens of points. The CI does the arguing, not the point
estimate.

Two evals exist and answer different questions: root `evaluate.py` scores the shipped hot channel on
all 300 labelled rows (n=11 emergencies, leave-cluster-out retrieval — fair, since the LLM never
trains), and is what `memo.md` quotes; `evals/evaluate.py` scores the 150 held-back rows (n=6),
because comparing against a model that *does* train requires a held-out half.

## Status

`predictions.csv` (300 rows) comes from the hot path via `backfill.py`, carrying the LLM's own
one-line reason per row: 229 routine / 61 same_day / 10 emergency. At the hard stop the OpenAI
account had hit a request cap and this shipped from the cold path; the key was replaced afterwards,
which tested the resumability claim for real — every previously scored body came back as a SQLite
cache hit and only the remainder was re-called.

That resumed run changed two things worth knowing. Safety rules now re-apply to cached LLM verdicts
rather than the stored floored one (`app/service.py`): rules are free and get edited often, LLM
calls cost money and don't, so a rule fix now takes effect with no re-classification run. And the
safety regexes were typo-brittle — `smell (of )?gas` never fires on *"smel gass"* — which the trap
set caught; they now tolerate repeated and dropped letters, checked against "gas hob", "Gas Safe
registered" and "gas safety certificate" all staying routine.

## With more time

Ranked by what would change the answer.

1. **Get an uncontaminated emergency number.** Label 50 more emergencies, preferring the messages
   the system is *least* confident about. Worth more than any modelling change. The override
   endpoint is how that happens for free in production.
2. **Fix `same_day` recall** (16/22 hot, 9/22 cold). Loss-of-service and contained leaks are the
   recurring pattern, and they are the messages that make a customer feel ignored.
3. **Redact PII before the API call.** Addresses, phone numbers and key-safe locations go to OpenAI
   in full today. Cheap to strip; removes a real objection rather than documenting one.
4. **Run it silently for 2–4 weeks** and count what it *would* have buried. No customer affected.
5. **A time-based split alongside the grouped one.** The gap between them is the honest production
   estimate, and it is not measured yet.
6. **Group the cold path's CV by template.** `ml/cold.py` de-dupes on exact normalised core, but 813
   near-duplicate pairs survive into its 5-fold CV, so `config.json`'s calibration is optimistic.
7. **Split promote from retrain.** `POST /retrain` promotes in the same call; a metric-gated
   auto-promote can silently degrade safety behaviour on a bad night of labels.
8. **Fuse the ML second opinion into the hot path** (`final = max(rules, llm, ml)`, disagreements
   surfaced as the review queue). Designed and both models exist, but they are only compared offline.
9. **Out-of-hours routing**, and **age-decay in the sort** — fixes starvation of old routine
   messages with arithmetic rather than a model call.
10. **Repeat-contact signal** ("third time I've chased") — needs the sender identity this data
    withholds.
11. **Scale seams**, all marked `# ponytail:` in code: O(n²) Jaccard dedup and full TF-IDF refit per
    label (fine to ~10k rows, MinHash beyond), sequential LLM calls in backfill, SQLite, and the
    cold path reading a CSV where production would read the corpus table.

## Known inconsistencies, stated rather than hidden

- Hot and cold have **two different `normalise` functions and two de-duplication routines**
  (`app/service.py` vs `ml/common.py`) that resolve label conflicts differently — hot drops
  contradictory clusters, cold escalates them. Should be one routine with two consumers.
- The de-escalation trap is handled only by the LLM prompt, so the offline fallback has no defence
  against it. The trap set confirms both channels pass the four de-escalation cases anyway, because
  the safety rules fire regardless of tone.
- `ml/preprocess.py` pseudo-labels 438 unlabelled rows by cosine kNN into the training set — 76% of
  `train.csv` — using the same nearest-neighbour mechanism the scoreboard calls the leakage exhibit.
  It abstains below 0.6 similarity and mints almost no emergencies, but it deserves a harder look
  than the clock allowed.
