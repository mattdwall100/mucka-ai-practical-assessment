"""Preprocess: unify both CSVs, split the 300 gold labels 50/50 (grouped so near-duplicate
templates never straddle the split), pseudo-label everything else by kNN cosine vote.

Run: python ml/preprocess.py   ->  data/processed/{combined.csv, train.csv} + printed report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from common import (PROCESSED, URGENCY_ORDER, build_vectorizer, core_hash,
                    load_holdout, load_messages, normalise)

K = 5
SIM_FLOOR = 0.60   # ponytail: hand-picked; tune against abstention/error on a labelled dev set
JACCARD_DUP = 0.6  # same threshold the twin-conflict analysis used

SEED = 0


def dup_groups(bodies):
    """Connected components of token-Jaccard > JACCARD_DUP. O(n^2) — fine for n=300."""
    sets_ = [set(normalise(b).split()) for b in bodies]
    parent = list(range(len(sets_)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sets_)):
        for j in range(i + 1, len(sets_)):
            inter = len(sets_[i] & sets_[j])
            if inter and inter / len(sets_[i] | sets_[j]) > JACCARD_DUP:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(sets_))])


def main():
    df = pd.concat([load_messages(), load_holdout()], ignore_index=True)
    df["core_hash"] = df["body"].map(core_hash)

    gold = df[df["gold_label"].notna()].copy()
    gold_idx = gold.index.to_numpy()
    groups = dup_groups(gold["body"].tolist())

    # 50/50 grouped + stratified split of the gold rows
    skf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
    test_pos, train_pos = next(iter(skf.split(gold, gold["gold_label"], groups)))
    train_idx, test_idx = gold_idx[train_pos], gold_idx[test_pos]
    assert set(groups[train_pos]).isdisjoint(groups[test_pos])

    df["label_source"] = "none"
    df.loc[train_idx, "label_source"] = "gold_train"
    df.loc[test_idx, "label_source"] = "gold_test"
    df["label"] = df["gold_label"]
    df["confidence"] = np.where(df["gold_label"].notna(), 1.0, np.nan)

    # conflicts inside gold twin-groups: same template, different label
    conflicts = []
    for g in np.unique(groups):
        sub = gold.iloc[np.flatnonzero(groups == g)]
        if sub["gold_label"].nunique() > 1:
            conflicts.append(sub[["message_id", "gold_label", "body"]])

    # pseudo-label the unlabelled rows against GOLD TRAIN only (gold_test stays untouched)
    vec = build_vectorizer().fit(df["body"])
    unl = df[df["gold_label"].isna()]
    train = df.loc[train_idx]
    S = cosine_similarity(vec.transform(unl["body"]), vec.transform(train["body"]))
    train_labels = train["gold_label"].to_numpy()

    labels, confs = [], []
    for row in S:
        top = np.argsort(row)[-K:]
        if row[top[-1]] < SIM_FLOOR:
            labels.append(None)
            confs.append(np.nan)
            continue
        w = {}
        for i in top:
            w[train_labels[i]] = w.get(train_labels[i], 0.0) + row[i]
        # tie (or near-tie) breaks toward the more urgent class
        winner = max(w, key=lambda c: (round(w[c], 6), URGENCY_ORDER.index(c)))
        labels.append(winner)
        confs.append(w[winner] / sum(w.values()))
    df.loc[unl.index, "label"] = labels
    df.loc[unl.index, "confidence"] = confs
    df.loc[unl.index[pd.notna(labels)], "label_source"] = "synthetic"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_cols = ["message_id", "received_at", "channel", "from_name", "body",
                "core_hash", "label", "label_source", "confidence", "source"]
    df[out_cols].to_csv(PROCESSED / "combined.csv", index=False)
    # training set = gold_train + confident synthetic, messages file ONLY
    # (holdout pseudo-labels are analysis-only: never train on guesses about the exam paper)
    train_out = df[(df["source"] == "messages")
                   & (df["label_source"].isin(["gold_train", "synthetic"]))]
    train_out[out_cols].to_csv(PROCESSED / "train.csv", index=False)

    # ---- report ----
    print(f"rows total {len(df)}  (messages {sum(df.source == 'messages')}, "
          f"holdout {sum(df.source == 'holdout')})")
    print("\nclass counts by label_source:")
    print(pd.crosstab(df["label_source"], df["label"].fillna("(none)")))
    n_abstain = int(df["label"].isna().sum())
    print(f"\nabstained (nearest neighbour < {SIM_FLOOR}): {n_abstain}")
    print(f"training rows written: {len(train_out)}")
    syn_em = ((df.label_source == 'synthetic') & (df.label == 'emergency')).sum()
    print(f"synthetic emergencies: {syn_em}  <- if ~0, the vote can't mint the minority class")
    print(f"\ngold twin-conflicts ({len(conflicts)} groups):")
    for c in conflicts:
        for _, r in c.iterrows():
            print(f"  [{r.gold_label:>9}] {r.body[:90]}")
        print()


if __name__ == "__main__":
    main()
