"""Hot channel vs cold model on the held-back gold test set (never trained on).

Hot = the production service path (few-shot + LLM + escalate-only rules), run through
eval.db so retrieval excludes each row's near-duplicate cluster (leakage guard) and
every call is cached/resumable.
Cold = the retrained TF-IDF+LR artifact named by data/config/current.txt.

Run: python evals/evaluate.py   (needs OPENAI_API_KEY for uncached hot rows)
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # app.*, backfill
sys.path.insert(0, str(ROOT / "ml"))   # common, cold

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.db import Repo, connect
from app.service import RANK, Retriever, apply_safety_floor, classify_message
from backfill import cluster_labelled, labelled_rows, seed
from backfill import load_messages as bf_load_messages
from cold import load_current
from common import PROCESSED, build_vectorizer


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw), min(1.0, c + hw)


def pct_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n}={k/n:.0%} (CI {lo:.0%}-{hi:.0%})"


def scoreboard_row(name, y_true, y_pred):
    parts = [f"{name:<15} acc {(y_true == y_pred).mean():.0%}"]
    for c in ("emergency", "same_day"):
        m = y_true == c
        parts.append(f"{c[:4]}-recall {pct_ci(int((y_pred[m] == c).sum()), int(m.sum()))}")
    r = y_true == "routine"
    parts.append(f"false-alarms {int((y_pred[r] != 'routine').sum())}/{int(r.sum())}")
    print("  ".join(parts))


def main():
    comb = pd.read_csv(PROCESSED / "combined.csv")
    test = comb[comb.label_source == "gold_test"].reset_index(drop=True)
    y = test["label"].to_numpy()
    print(f"gold_test: {len(test)} rows {test['label'].value_counts().to_dict()}")

    # ---- hot channel ----
    repo = Repo(connect("eval.db"))
    retriever = Retriever()
    seed(repo, retriever)
    labelled = labelled_rows(bf_load_messages())
    group_of = {m["message_id"]: g for g, mem in cluster_labelled(labelled) for m in mem}
    by_id = {m["message_id"]: m for m in labelled}
    hot = []
    try:
        if os.environ.get("EVAL_SKIP_HOT"):  # fast offline run: skip the LLM channel entirely
            raise RuntimeError("EVAL_SKIP_HOT set")
        for i, mid in enumerate(test["message_id"], 1):
            hot.append(classify_message(repo, retriever, by_id[mid],
                                        exclude_group=group_of[mid])["urgency"])
            if i % 25 == 0:
                print(f"... hot {i}/{len(test)}")
        hot = np.array(hot)
    except Exception as e:  # no key/credits: report cold + baselines; rerun resumes from cache
        print(f"hot channel unavailable after {len(hot)} rows ({type(e).__name__}) "
              "- scoreboard shows cold + baselines only")
        hot = None

    # ---- cold model ----
    model, cfg = load_current()
    cold_pred = model.predict(test["body"])
    print(f"cold artifact: {cfg['version']}, trained on {cfg['train_rows']} rows")

    # ---- baselines ----
    always = np.full(len(test), "routine")
    rules = np.array([apply_safety_floor(b, "routine")[0] for b in test["body"]])
    train = comb[comb.label_source == "gold_train"]
    vec = build_vectorizer().fit(comb["body"])
    S = cosine_similarity(vec.transform(test["body"]), vec.transform(train["body"]))
    nn = train["label"].to_numpy()[S.argmax(1)]  # the leakage exhibit

    print("\n--- scoreboard on gold_test (grouped split: no near-twin spans train/test) ---")
    # what ml/predict.py actually ships: cold model + escalate-only safety floor
    fused = np.array([apply_safety_floor(b, p)[0] for b, p in zip(test["body"], cold_pred)])
    systems = [("always-routine", always), ("rules-only", rules),
               ("1NN-cosine", nn), ("cold TFIDF+LR", cold_pred),
               ("cold+rules", fused)]
    if hot is not None:
        systems.append(("hot LLM+rules", hot))
    for name, pred in systems:
        scoreboard_row(name, y, pred)

    # ---- hot vs cold: disagreement = the needs_human queue ----
    if hot is not None:
        dis = hot != cold_pred
        print(f"\nhot vs cold disagree on {int(dis.sum())}/{len(test)} rows "
              f"(~{dis.mean() * 90:.0f} msgs/day sent to a human at 90/day)")
        print(pd.crosstab(pd.Series(hot, name="hot"), pd.Series(cold_pred, name="cold")))
        print("\ndisagreement rows (true label first):")
        for i in np.flatnonzero(dis):
            print(f"- true={y[i]:<9} hot={hot[i]:<9} cold={cold_pred[i]:<9} {test['body'].iloc[i][:110]}")

    # ---- what would have been buried ----
    final, channel = (hot, "hot") if hot is not None else (fused, "cold+rules")
    down = [i for i in range(len(test)) if RANK[final[i]] < RANK[y[i]]]
    print(f"\nDOWNGRADES by {channel} channel ({len(down)}):")
    for i in down:
        print(f"- true={y[i]} pred={final[i]}: {test['body'].iloc[i][:150]}")

    print("\ncaveat: every number above measures agreement with one busy labeller "
          "(9/225 near-twin label conflicts in the gold set), not ground truth. "
          "emergency n=6: the interval spans tens of points.")


if __name__ == "__main__":
    main()
