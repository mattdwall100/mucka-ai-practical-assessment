# Evaluation

## What I measured, and why

**Recall on the urgent classes, with the false-alarm rate stated as its price.** Accuracy is
useless here: predicting `routine` for everything scores 81% on the
held-out half below, and 82% across all 300 labelled rows (the figure `memo.md` uses). The founder's real question is
*"how often does this bury something that needed me now?"* — that is recall on `emergency` and
`same_day`. A missed gas leak is a fire; a wrongly escalated routine message costs ten seconds.

Everything else follows from that asymmetry:

| measured | why |
|---|---|
| Per-class recall + **Wilson CIs** | n=11 emergencies. The interval does the arguing, not the point estimate. |
| **False alarms**, always beside recall | Recall alone is gamed by flagging everything. |
| The **downgrade list, full text** | The output that actually informs a decision. A confusion matrix doesn't tell you *which* households. |
| **Baselines** (always-routine, rules-only, 1-NN) | Proves the system beats trivial alternatives. 1-NN is the leakage exhibit. |
| **Hot vs cold agreement** | Where they disagree is the `needs_human` queue, and its size is an operating cost. |
| **Trap set** | Held-out adversarial cases. Real-world robustness, not corpus agreement. |

Split is **grouped by near-duplicate cluster**, not random: 174/300 holdout messages have a
>0.7-Jaccard twin, so a random split reports memorisation.

## The code

| file | what it produces |
|---|---|
| `evaluate.py` | Hot channel, all 300 labelled rows, leave-cluster-out retrieval. n=11. |
| `evals/evaluate.py` | Hot vs cold vs 3 baselines on the 150-row held-out half. n=6. |
| `evals/traps.py` + `traps.csv` | 22 hand-written adversarial messages. |
| `ml/cold.py` | Threshold calibration to the alert budget. |

Two evals because the LLM never trains (all 300 is fair to it) but the cold model does (needs a
held-out half).

## The numbers

**Shipped hot channel — `python evaluate.py`, all 300 labelled:**

```
                routine  same_day  emergency
routine             231        14          0
same_day              4        37          3
emergency             0         0         11
```

| | |
|---|---|
| emergency recall | **11/11** † (honest figure: **7/11**) |
| same_day recall | 37/44 = 84% |
| false alarms | 14/245 = 6% |
| buried (urgent → routine) | 4, two of which are label noise (a VAT query, a weeks-old certificate) |

**Hot vs cold — `python evals/evaluate.py`, 150 held-out:**

```
always-routine  acc 81%  emer 0/6              same 0/22               false-alarms 0/122
rules-only      acc 81%  emer 6/6 †            same 0/22               false-alarms 6/122
1NN-cosine      acc 83%  emer 0/6              same 11/22              false-alarms 8/122
cold TFIDF+LR   acc 86%  emer 0/6              same 9/22  (CI 23-61%)  false-alarms 2/122
hot LLM+rules   acc 90%  emer 6/6 †            same 16/22 (CI 52-87%)  false-alarms 9/122
```

At the shipped operating point (0.66, ~11 flags/day): emergency 6/6 †, same_day 6/22, 12/28
urgent, 7 false flags/122 routine. All six emergencies score *below* 0.66 — every one is caught by
a rule, not the model.

**Trap set — `python evals/traps.py`:**

```
cold  exact 16/22   under-ranked 4   over-ranked 2
hot   exact 20/22   under-ranked 0   over-ranked 2
```

## What the numbers say

1. **86% accuracy and 0/6 emergency recall in the same row.** The most accurate model never
   catches an emergency — with ~7 emergency training rows it cannot learn the class. That is the
   argument for the rules + LLM architecture, made by measurement.
2. **The LLM earns its cost on `same_day` (16/22 vs 9/22), not emergencies.** Those are caught by
   free deterministic rules that cannot be talked out of firing.
3. **Hot under-ranks nothing on the trap set.** Both its errors over-rank — the preferred direction.
4. **Hot and cold disagree on 25/150** (~15/day), almost all same-day work cold missed. Routing
   disagreements to a human is a cheap safety net.

## † Contamination, stated plainly

The `vulnerable_no_service` rule (vulnerable occupant + total loss of heat → emergency) was written
**after reading the four emergency misses in the test set**. Every emergency figure marked †
is fitted, not predicted — including the rules-only baseline, which clears 6/6 only because of it.
Honest pre-rule numbers: **7/11** and **2/6**. The rule is defensible on
its own terms — a three-week-old baby with no heat is an emergency, and saying so needs no training
data — but the number it produces is not evidence. Only a fresh labelled sample fixes it.

## Limits

- **n=11 emergencies.** Every interval spans tens of points.
- **The labels disagree with themselves** — 9 of 225 near-duplicate pairs. This is agreement with
  one busy labeller, not truth, and it caps achievable accuracy.
- **No time-based split yet.** Grouped CV predicts the holdout but overstates production.
- **Intent is unevaluated** — no labels exist for it.
