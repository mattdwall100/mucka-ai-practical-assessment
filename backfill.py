"""Backfill: seed the corpus from messages.csv, classify holdout.csv through the
same service path as the live API, write predictions.csv. Idempotent — a crashed
run resumes for free because every already-scored body is a cache hit.

Run: python backfill.py   (needs OPENAI_API_KEY)
"""
import glob

import pandas as pd

from app.db import Repo, connect
from app.service import LABELS, Retriever, body_hash, classify_message, normalise

MESSAGES_CSV = glob.glob("artifacts/*messages*.csv")[0]
HOLDOUT_CSV = glob.glob("artifacts/*holdout*.csv")[0]


def load_messages():
    df = pd.read_csv(MESSAGES_CSV)
    # explicit format per file — never dayfirst inference (the two CSVs differ)
    df["received_at"] = pd.to_datetime(
        df["received_at"], format="%d/%m/%Y %H:%M:%S").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df


def load_holdout():
    df = pd.read_csv(HOLDOUT_CSV)
    df["received_at"] = pd.to_datetime(
        df["received_at"], format="%Y-%m-%d %H:%M").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df


def labelled_rows(df):
    return [r for r in df.to_dict("records") if r.get("urgency") in LABELS]


def cluster_labelled(rows):
    """Union-find over token-Jaccard > 0.6: near-duplicate clusters.
    Returns [(dup_group, [rows])]. Reused by evaluate.py as the leakage guard.
    # ponytail: O(n^2) over 300 rows; MinHash if the corpus reaches 10k.
    """
    toks = [set(normalise(r["body"]).split()) for r in rows]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            inter = len(toks[i] & toks[j])
            if inter and inter / len(toks[i] | toks[j]) > 0.6:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    return [(g, [rows[i] for i in members])
            for g, (_, members) in enumerate(sorted(groups.items()))]


def seed(repo, retriever):
    """messages.csv -> messages table; de-conflicted labelled rows -> corpus."""
    df = load_messages()
    records = df.to_dict("records")
    for r in records:
        repo.upsert_message({**r, "body_hash": body_hash(r["body"])})

    labelled = labelled_rows(df)
    dropped = 0
    for g, members in cluster_labelled(labelled):
        if len({m["urgency"] for m in members}) > 1:
            dropped += len(members)  # contradictory twins: worse than no shot — drop the cluster
            continue
        rep = members[0]  # agreeing near-dups: one representative is enough
        repo.add_corpus_row({
            "message_id": rep["message_id"], "body": rep["body"],
            "body_hash": body_hash(rep["body"]), "urgency": rep["urgency"],
            "dup_group": g, "source": "seed",
        })
    retriever.fit(repo.corpus_rows())
    print(f"seeded: {len(records)} messages, corpus {repo.corpus_size()} "
          f"from {len(labelled)} labels ({dropped} dropped as contradictory near-dups)")


def main():
    repo = Repo(connect("app.db"))
    retriever = Retriever()
    seed(repo, retriever)

    records = load_holdout().to_dict("records")
    # ponytail: sequential LLM calls (~300); ThreadPoolExecutor(8) if the clock demands
    for n, r in enumerate(records, 1):
        classify_message(repo, retriever, r)
        if n % 25 == 0:
            print(f"classified {n}/{len(records)}")

    rows = [{"message_id": r["message_id"], **{k: c[k] for k in ("urgency", "score", "reason")},
             "source": "hot"}  # so the file says which system produced it
            for r in records if (c := repo.get_classification(body_hash(r["body"])))]
    pd.DataFrame(rows).to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
