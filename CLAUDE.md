# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A timeboxed practical assessment for an ML Engineering Intern role at **Mucka.ai** (AI business
operating system for UK trades businesses — plumbers, gas engineers, HVAC).

**Task:** for each inbound customer message (SMS / WhatsApp / transcribed voicemail), decide how
urgently a human needs to see it. Then give an honest assessment of whether it should go in front
of a customer.

The brief is deliberately underspecified — urgency levels, the definition of "good", the metric,
and the method are all ours to choose. **The assessment is on the reasoning, not the accuracy
score.** They stated they are reading for three things:

1. Whether the evaluation would survive contact with reality.
2. Whether we noticed the things they didn't tell us about.
3. Whether the memo would actually help a busy, non-technical person make a decision.

Brief: [artifacts/Mucka — ML Engineering Intern — Practical Assessment.md](artifacts/Mucka%20%E2%80%94%20ML%20Engineering%20Intern%20%E2%80%94%20Practical%20Assessment.md)

**Hard stop: 15:30 on 2026-08-20.** Stop building then and write the "what I'd do with more time"
section. Finishing early is explicitly not penalised. Prefer a complete, honest, small system over
an incomplete clever one.

## Deliverables (all four are graded)

| File | Contents |
|---|---|
| `README.md` | How to run from a clean checkout; decisions and their rationale; "with more time". |
| `predictions.csv` | One row per holdout message: `message_id` + predicted urgency (+ useful extras such as score and reason). |
| Evaluation | The numbers **and** the code that produced them, plus why those metrics. |
| `memo.md` | One page to a non-technical founder who runs a plumbing business. Ship it or not, and what would make us confident. |

## Data — verified facts, not assumptions

Both CSVs are in [artifacts/](artifacts/). Every number below was measured; re-measure rather than
trusting from memory if it matters.

- **1,252 messages, only 300 labelled (24%).** Labels: `routine` 245, `same_day` 44,
  **`emergency` 11**. The holdout is 300 unlabelled messages.
- **Eleven emergency examples is the central constraint of this task.** Any per-class number for
  emergency carries an enormous confidence interval. Report intervals or don't report the number.
- **The label set is not given in the brief** — `routine`/`same_day`/`emergency` are inferred from
  the labelled rows. Whether that taxonomy is the right one is a decision to defend, not inherit.
  There is a real case for a fourth level (out-of-hours / life-safety) and for "needs a human at
  all" being a separate axis from "how fast".
- **The labels are inconsistent.** 9 of 225 near-duplicate labelled pairs (token Jaccard > 0.6)
  disagree — e.g. *"no hot water since this morning, heating seems fine"* appears as both
  `same_day` and `routine`; *"radiators stone cold upstairs"* likewise. This is the office manager
  labelling by mood, time of day, or who was calling. **It puts a ceiling on achievable accuracy
  (~4% irreducible on near-identical text) and it is one of the "things they didn't tell us".**
- **Near-duplicate leakage: 174 of 300 holdout messages have a >0.7-Jaccard twin in `messages`.**
  Bodies are built from a template core plus spliced-in noise (greetings, sign-offs, irrelevant
  fragments like *"bungalow, single storey"*, typos such as `teh`, ALL-CAPS, `[inaudible]`).
  A random train/test split will therefore report a flattering, wrong number. **Group by
  normalised body core, or split by time, before believing any CV score.**
- **The two files use different date formats:** `messages` is `DD/MM/YYYY HH:MM:SS`,
  `holdout` is `YYYY-MM-DD HH:MM`. Parse per file. Do not let a parser silently coerce.
- **The two files are one dataset of 1,552 messages, randomly split — not two sources.** Message
  IDs run `m00001`–`m01552` with the union of both files covering all 1,552 with no gaps and no
  overlap. Channel mix (47/35/18 vs 49/34/17), median body length (103 vs 107), monthly volume and
  out-of-hours rate all match. So the holdout is an IID sample, **not** a future period.
  Consequence: a random-split CV on `messages` will *predict the holdout score well* while
  *overstating production performance*, because production is temporal. Report a time-based split
  and a template-grouped split as well — the gap between them is the honest finding.
- **`from_name` is not an identity.** 40 first names × 20 surnames recombined; 185 names appear in
  both files by coincidence. There is no repeat-customer signal to extract, and building one would
  be fitting noise. Worth saying out loud in the memo: in the *real* system, customer history
  (is this the landlord with 40 properties? did they message twice in an hour?) is probably the
  strongest available signal, and this dataset deliberately withholds it.
- **26% of messages (324/1252) arrive outside 08:00–18:00.** Urgency and actionability are not the
  same thing: an emergency at 21:40 needs a different route than one at 10:00. `received_at` is
  available at inference time and is fair game as a feature or as routing logic downstream.
- Messages carry **PII in the body** — phone numbers, addresses, key-safe locations
  (*"the neighbours have a spare key"*). Don't paste raw bodies into anything external; if an LLM
  API is used, say so and mind what is sent.
- **Multi-intent messages are common**: one body can be a compliment, a billing question, and a
  leak report at once. Single-label classification loses this; note it rather than pretending.
- **The most dangerous messages in the dataset are unlabelled, and labelling is random.** The
  labelled rate inside each hazard family (21–50%) tracks the 24% base rate, so the 300 labels are
  a random sample — *not* the office manager labelling the urgent ones. Two consequences:
  **(a) unlabelled ≠ routine** — anyone who trains on the 952 unlabelled rows as negatives is
  teaching the model that gas leaks are fine; **(b) whole hazard families have zero label support:**

  | Hazard family | in `messages` | labelled as | in `holdout` |
  |---|---|---|---|
  | Carbon monoxide alarm going off, house evacuated | 1 | **nothing** | 2 |
  | Strong gas smell *indoors* (under stairs, landing, bathroom) | 5 | **nothing** | 2 |
  | Burning smell + lights flickered, killed at the fuse box | 1 | **nothing** | 1 |
  | Boiler hissing and smelling hot | 12 | `same_day` ×3 | 3 |
  | Gas smell *outdoors* by the meter box | 3 | `emergency` ×2 | 0 |
  | Burst tank / water through the ceiling | 8 | `emergency` ×4 | 3 |

  Read the top three rows carefully: *"Carbon monoxide alarm has gone off twice this morning.
  Everyone's outside. What do we do"* has **no label anywhere in the dataset** and appears twice in
  the holdout. A purely supervised model cannot learn it. Meanwhile the only labelled gas examples
  are the *outdoor* variant, which is less severe than the unlabelled indoor ones.
- **At least one label looks simply wrong.** A boiler that is hissing and smells hot is a
  gas-appliance fault people are told to evacuate for; it is labelled `same_day` all three times it
  is labelled. Do not treat these labels as ground truth — treat them as *one busy person's
  historical behaviour*, which is the thing the product is supposed to improve on.
- **A rule layer that can only escalate is therefore not a shortcut, it is the correct design.**
  The highest-severity categories have zero or actively misleading supervision. This also reframes
  the product question: for gas and CO the right first action is not "put it in the queue" but
  "reply now with the National Gas Emergency number (0800 111 999) and evacuation advice" — the
  plumbing firm is not even the correct first responder.
- Signals worth being deliberate about, with measured counts across both files: gas smell/leak 11,
  carbon monoxide 3, leak/flood 34, vulnerable occupant (baby, elderly, oxygen) 40, explicit
  de-escalation ("not urgent", "no rush", "whenever") 91. The last one is a trap for bag-of-words
  models — *"radiator valve is stiff, not urgent"* contains "radiator" and reads urgent on keywords.

## Evaluation stance

Accuracy is the wrong headline metric here — predicting `routine` for everything scores 82% on the
labelled set. The cost of errors is wildly asymmetric: a missed gas leak or burst tank is a flood,
a fire, or a person; a routine message escalated by mistake costs the office manager ten seconds.

Whatever is built, the evaluation must answer the question the founder actually has: **"how often
does this bury something that needed me now?"** Prefer recall on the urgent classes, with the
false-alarm rate stated alongside it as the price paid, and confidence intervals that admit
n = 11. A confusion matrix of what got downgraded, with the actual message text, is worth more to
this audience than a macro-F1.

## Conventions for code written here

- Python 3.14 (`python`, not `python3`, on this Windows box). `pip install -r requirements.txt`.
  Verified working on this machine: scikit-learn 1.9.0, pandas 3.0.5 (3.14 wheels exist — checked
  early, because discovering a missing wheel at 15:00 would have cost the deliverable).
- Keep it to a handful of scripts that run top-to-bottom and print their numbers. No framework,
  no package layout, no config system. The reader has 30 minutes and wants to see the reasoning.
- Every script must run from a clean checkout with no manual steps and no network, or the README
  must say plainly what it needs.
- The artifacts directory holds the given inputs; treat it as read-only.
- Deliberate shortcuts get a `# ponytail:` comment naming the ceiling and the upgrade path, so
  they end up in the README's "with more time" instead of being quietly forgotten.

## Working notes

Running decision log lives in [NOTES.md](NOTES.md) — append to it as decisions get made. It is the
raw material for the README rationale and `memo.md`; writing those from scratch at 15:00 is how the
reasoning gets lost.
