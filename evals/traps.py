"""Adversarial trap set: hand-written failure modes, held out from everything.

Not a score to optimise — a list of the ways this system is known to break. Each row
targets one documented weakness (de-escalation language, transcription truncation,
typos, buried multi-intent, vulnerable occupant with no alarm word, 'gas' in the
wrong sense). Both channels are scored; disagreements are the interesting rows.

Run: python evals/traps.py           (hot needs OPENAI_API_KEY)
     TRAPS_SKIP_HOT=1 python evals/traps.py    (cold only, offline)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

import numpy as np
import pandas as pd

from app.db import Repo, connect
from app.service import RANK, Retriever, apply_safety_floor, classify_message
from backfill import cluster_labelled, labelled_rows
from backfill import load_messages as bf_load_messages
from cold import load_current


def main():
    traps = pd.read_csv(ROOT / "evals" / "traps.csv")
    model, cfg = load_current()
    classes = list(model.classes_)

    # cold channel = what ml/predict.py ships: model argmax + escalate-only safety floor
    base = model.predict(traps["body"])
    cold = np.array([apply_safety_floor(b, p)[0] for b, p in zip(traps["body"], base)])

    hot = None
    if not os.environ.get("TRAPS_SKIP_HOT"):
        try:
            repo = Repo(connect("eval.db"))          # same cache as the eval run
            retriever = Retriever()
            retriever.fit(repo.corpus_rows())
            # traps are new text, so no cluster to exclude — nothing to leak from
            hot = np.array([classify_message(repo, retriever,
                                             {**r, "message_id": f"trap_{r['trap_id']}"})["urgency"]
                            for r in traps.to_dict("records")])
        except Exception as e:
            print(f"hot channel unavailable ({type(e).__name__}) — cold only")

    exp = traps["expected"].to_numpy()
    channels = [("cold", cold)] + ([("hot", hot)] if hot is not None else [])
    print(f"\n--- trap set: {len(traps)} hand-written adversarial messages ---")
    for name, pred in channels:
        exact = int((pred == exp).sum())
        # under-ranking is the failure that matters; over-ranking costs ten seconds
        under = int(sum(RANK[p] < RANK[e] for p, e in zip(pred, exp)))
        over = int(sum(RANK[p] > RANK[e] for p, e in zip(pred, exp)))
        print(f"{name:<5} exact {exact}/{len(traps)}   under-ranked {under}   over-ranked {over}")

    print("\nper-trap (under-ranked marked MISS — the ones that would get buried):")
    for i, r in traps.iterrows():
        cells = "  ".join(f"{n}={p[i]:<9}" for n, p in channels)
        flag = "  MISS" if any(RANK[p[i]] < RANK[r.expected] for _, p in channels) else ""
        print(f"[{r.trap_kind:<21}] want={r.expected:<9} {cells}{flag}")
        if flag:
            print(f"      {r.body[:120]}")

    if hot is not None:
        dis = np.flatnonzero(hot != cold)
        print(f"\nhot/cold disagree on {len(dis)}/{len(traps)} traps "
              f"(in production these route to a human):")
        for i in dis:
            print(f"- want={exp[i]:<9} hot={hot[i]:<9} cold={cold[i]:<9} {traps['body'].iloc[i][:90]}")


if __name__ == "__main__":
    main()
