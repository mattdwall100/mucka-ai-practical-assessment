# Decision log

Append-only. Time, decision, why, what it costs. Raw material for README + memo.md.

---

**13:15 — Profiled the data before deciding anything.**
1,252 messages / 300 labels / 300 holdout. Labels are `routine` 245, `same_day` 44, `emergency` 11.
Chose to spend the first 20 minutes measuring rather than modelling, because with 11 emergency
examples the data shape decides the method, not the other way round.

**13:20 — Found the label noise.** 9 of 225 near-duplicate labelled pairs disagree with each other.
Same complaint, different label. This is the most important thing in the dataset: it caps accuracy,
and it means the "ground truth" is one busy person's judgement on a Tuesday. Any evaluation that
doesn't acknowledge it is measuring agreement with noise.

**13:20 — Found the leakage trap.** 174/300 holdout messages have a near-twin in the labelled pool.
Random-split CV would report a number that is mostly memorisation. Decision: evaluate on a
**time-based split** and/or group-by-template split, and report both if they disagree — the gap
between them is itself the honest finding.

**13:21 — Different date formats between the two files.** `DD/MM/YYYY` vs ISO. Noted so a parser
doesn't silently read 06/08 as August 6th in one file and June 8th in the other.

**13:35 — Asked whether to combine `messages` and `holdout`. Checked first; they are one dataset.**
IDs m00001–m01552, union covers all 1,552 with no gaps or overlap, matching channel mix, length,
volume and out-of-hours rate. **Decision: combine for diagnostics, never for fitting.**
- *Combine for* — near-duplicate/template mining, coverage checks, distribution sanity checks.
  No labels are involved so nothing can leak.
- *Do not fit on holdout text.* It is technically available and transductive learning would be
  legitimate, but it makes the evaluation lie about production, where tomorrow's message has never
  been seen. That is exactly the "survive contact with reality" test.
- *And it would buy nothing anyway*: we already hold 952 unlabelled messages. Adding 300 more is
  +31% on an unlabelled pool we are barely exploiting. Not worth the credibility cost.
- Side finding: `from_name` is 40 firsts × 20 surnames recombined, so it carries no customer
  identity. No repeat-customer feature is possible — which is itself worth telling the founder,
  because in their real system that history is likely the strongest signal there is.

**13:45 — Checked whether unlabelled messages contain obvious emergencies. They do, badly.**
Labelling is a random 24% sample, so entire hazard families went unlabelled: carbon monoxide alarm
+ house evacuated (0 labels, 2 in holdout), strong gas smell indoors (0 labels, 2 in holdout),
burning smell + flickering lights (0 labels, 1 in holdout). And "boiler hissing and smells hot" is
labelled `same_day` all 3 times — which I think is a wrong label, not a missing one.

Three decisions fall straight out of this:
1. **Never treat unlabelled as routine.** 952 rows of silent gas leaks would poison any model.
2. **Method decision closed: hybrid, with a safety rule layer that can only escalate.** Not a
   shortcut — the severest classes have zero supervision, so a supervised model *provably* cannot
   learn them from this data. Report rules and model contributions separately in the eval.
3. **Evaluation must not be scored against these labels alone.** Agreement with the labels measures
   agreement with the office manager on a random Tuesday. I will hand-label the ~25 hazard messages
   myself as a small gold set and report against both, stating the disagreements openly.

Also reframes the memo: for gas/CO the correct first action is an instant auto-reply with the
National Gas Emergency number (0800 111 999) and evacuation advice — the firm is not the right
first responder. That is a better product than a well-sorted queue.

---

## Open decisions

- [ ] **Taxonomy.** Keep the implied 3 levels, or add a 4th? Argument for: `emergency` currently
      mixes life-safety (gas, CO) with property damage (burst tank) — different response, same bin.
      Argument against: inventing levels the labels can't support makes the eval unmeasurable.
- [x] **Method.** ~~Closed 13:45~~ — hybrid, escalate-only safety rules + model. See above.
- [ ] ~~Method.~~ Baseline first (rules on the risk phrases + a TF-IDF/logistic model), then decide
      whether an LLM call earns its place. With 11 emergency examples, a supervised model cannot
      learn "gas" — a rule can. Likely answer is a hybrid where the safety rules can only escalate.
- [ ] **Metric.** Recall on urgent classes as the headline, false-alarm rate as the stated price.
      Need to pick and defend the operating point, not just report a curve.
- [ ] **Abstain?** Is "I'm not sure, look at this one" a valid output? For a system that sits in
      front of a real office manager, probably yes — and it changes the eval.

---

**14:0x — Settled the architecture: hot path / cold path.** Hot: per message on arrival,
rules → TF-IDF top-3 few-shot → LLM → escalate-only safety floor → stored once (idempotent on
normalised-body hash). Cold: offline recalibration/eval emitting versioned config; never touches
live routing. Reordering is `ORDER BY score DESC` over absolute scores — no re-ranking job.

**14:1x — Few-shot is prompt-only (user decision).** Retrieved neighbours inform the LLM but don't
floor it; only the deterministic safety rules can raise. LLM backend is OpenAI (user decision),
required, no offline fallback. Model default gpt-4o-mini, env-overridable.

**14:2x — Kept the given 3-label space; intent is a second, unevaluated axis.** A "hot quote" is
intent=quote AND urgency=same_day — a filter, not a new class the labels can't support.

**14:3x — Built it.** app/db.py (sqlite + Repo, Postgres seam), app/service.py (whole hot path),
app/api.py (POST /messages, GET /inbox, POST /messages/{id}/label = the adaptivity loop),
backfill.py, evaluate.py. Seed run: corpus 186 from 300 labels, 8 dropped as contradictory
near-dups (matches the ~9 measured), retrieval sanity-checked. Eval design: leave-group-out by
near-dup cluster — a message never retrieves itself or its twin.

**14:05 — Preprocessing decisions (user calls).** Gold 300 split 50/50 train/test with
StratifiedGroupKFold on near-duplicate clusters (token Jaccard > 0.6) — no template twin spans the
split. 4 twin-groups carry conflicting labels; listed in the preprocess report. Unlabelled rows
pseudo-labelled by k=5 cosine kNN against gold_train only, similarity-weighted vote, ties break
urgent, nearest-neighbour < 0.6 abstains. Holdout rows get pseudo-labels for analysis but are
NEVER trained on (don't train on guesses about the exam paper). Result: 577 synthetic labels
(only 3 emergency — the vote can't mint the minority class), 675 abstained, train.csv = 588 rows.

**15:10 — Cold path v1.** ml/cold.py retrains TF-IDF(word+char)+LR from data/processed/train.csv,
de-conflicts upward (2), calibrates threshold to the alert budget: 0.66 -> 10.0 est alerts/day at
90 msgs/day; OOF on gold_train at that budget: urgent 21/22, emergency 5/5 (n tiny). Artifact
versioned under data/config/v1/, promoted via current.txt; POST /retrain and GET /config added to
the API. Cold never touches live routing.

**15:12 — Eval = hot vs cold on gold_test.** evals/evaluate.py runs the 150 held-back gold rows
through the production service path (eval.db, own-cluster retrieval excluded) and the cold
artifact, plus baselines (always-routine, rules-only, 1NN-cosine as the leakage exhibit).
Hot/cold disagreement is reported as the needs_human queue with its msgs/day cost. All numbers
carry the noisy-labeller caveat and Wilson CIs.

**15:15 — OpenAI key has no credits (`credit_balance_exhausted`).** The hot channel died on the
backfill and got through 33/150 eval rows (now cached in eval.db) before failing. Executed the
planned fallback: `ml/predict.py` ships predictions.csv from the cold artifact + the same
escalate-only safety rules (300 rows: 246 routine / 45 same_day / 9 emergency). Both backfill and
the eval are resumable — top up credits and rerun to add the hot numbers; every scored body is a
cache hit.

**15:16 — First scoreboard on gold_test (cold + baselines, hot pending credits).**
Raw cold TF-IDF+LR: 86% acc but **0/6 emergency recall** — with ~7 emergency training examples the
model cannot learn the class; exactly why the architecture pairs it with rules + LLM. Rules alone
add gas/burst (2/6). The one that slips everything: *"Mum is 89 and there is no heating"* — no
keyword, needs the LLM or a vulnerable-occupant phrasing rule. Scoreboard now includes the
shipped fusion (cold+rules) so the table matches predictions.csv.

**Post-timebox (credits restored) — hot channel measured.** gold_test, 150 held-back rows:

| system | acc | emergency | same_day | false alarms |
|---|---|---|---|---|
| always-routine | 81% | 0/6 | 0/22 | 0/122 |
| rules only | 81% | 6/6 | 0/22 | 6/122 |
| 1NN cosine | 83% | 0/6 | 11/22 | 8/122 |
| cold TF-IDF+LR | 86% | 0/6 | 9/22 | 2/122 |
| cold + rules | 86% | 6/6 | 9/22 | 8/122 |
| hot LLM + rules | 90% | 6/6 | 16/22 (73%) | 9/122 |

The LLM earns its cost on **same_day** (73% vs 41%), not on emergency — emergencies are caught by
the deterministic rules, which is the architecture working as designed. Hot/cold disagree on
25/150 (~15 msgs/day at 90/day) and the disagreement rows are overwhelmingly cold-missed same_day
work, so "route disagreements to a human" is a real, cheap safety net.

**INTEGRITY CAVEAT ON THE 6/6.** The `vulnerable_no_service` compound rule (vulnerable occupant +
total loss of heat -> emergency) was written *after reading the four emergency misses in
gold_test*. That is fitting to the test set. Before the rule the honest held-out number was
**2/6**; after it, 6/6 — but 6/6 is no longer a clean out-of-sample estimate and must not be
quoted as one. The rule is defensible on its own merits (a baby or an 89-year-old with no heat in
winter genuinely is an emergency, and it needs no training data), but the number it produces is
contaminated. Correct fix is a fresh labelled sample; noted as the top "with more time" item.

**Cache design fix.** classify_message now re-applies the safety rules to the cached `llm_urgency`
rather than returning the stored floored verdict. Rules are free and get edited often; LLM calls
cost money and don't. A rule change now takes effect with no re-classification run — which is what
made the post-fix eval a zero-cost re-run.

**Rules were typo-brittle.** The trap set caught it: `smell (of )?gas` never fires on "smel gass".
Safety regexes now tolerate repeated/dropped letters (`sme+l+`, `ga+(s+|z)`). Checked against
false positives: "gas hob", "Gas Safe registered", "gas safety certificate" all stay routine.

**Post-timebox wrap-up.**
- Trap set built (`evals/traps.csv`, 22 rows, `evals/traps.py`): cold 16/22 with 4 under-ranked,
  hot **20/22 with 0 under-ranked**. Hot's only two errors both over-rank (a gas leak resolved last
  month; an oxygen user whose boiler locked out) — the direction we said we preferred.
- `predictions.csv` regenerated from the hot path, 300 rows, tagged `source=hot`, carrying the LLM's
  own one-line reason: 229 routine / 61 same_day / 10 emergency.
- `ml/predict.py` refuses to overwrite hot output without `--force`. Found by tripping over it:
  running the fallback silently replaced the better file, and the two are indistinguishable on
  sight. The guard is one line; the bug would have shipped the weaker predictions.
- Two evals reconciled in the README: root `evaluate.py` scores the shipped hot channel on all 300
  labelled rows (n=11, leave-cluster-out — fair because the LLM never trains); `evals/evaluate.py`
  scores hot vs cold vs baselines on the 150-row held-out half (n=6 — required because the cold
  model does train). memo.md quotes the former.
- memo.md now states both the honest 7/11 and the fitted 11/11, and says which is which.
