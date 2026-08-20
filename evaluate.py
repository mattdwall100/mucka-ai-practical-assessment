"""Leave-group-out evaluation on the 300 labelled messages, through the SAME
service path as production. Uses its own eval.db: retrieval here excludes each
message's near-duplicate cluster (the leakage guard), so its cache must never
pollute the production one. Resumable like backfill.

Run: python evaluate.py   (needs OPENAI_API_KEY)
"""
from app.db import Repo, connect
from app.service import LABELS, RANK, Retriever, classify_message
from backfill import cluster_labelled, labelled_rows, load_messages, seed


def main():
    repo = Repo(connect("eval.db"))
    retriever = Retriever()
    seed(repo, retriever)  # identical corpus build to production

    labelled = labelled_rows(load_messages())
    group_of = {m["message_id"]: g for g, members in cluster_labelled(labelled) for m in members}

    preds = []
    for n, r in enumerate(labelled, 1):
        out = classify_message(repo, retriever, r, exclude_group=group_of[r["message_id"]])
        preds.append((r, out))
        if n % 25 == 0:
            print(f"... {n}/{len(labelled)}")

    # confusion matrix
    cm = {(t, p): 0 for t in LABELS for p in LABELS}
    for r, out in preds:
        cm[(r["urgency"], out["urgency"])] += 1
    w = max(len(l) for l in LABELS) + 2
    print("\nconfusion matrix (rows = office manager, cols = system):")
    print(" " * w + "".join(l.rjust(w) for l in LABELS))
    for t in LABELS:
        print(t.rjust(w) + "".join(str(cm[(t, p)]).rjust(w) for p in LABELS))

    # per-class recall — emergency headlined WITH its caveat
    print()
    for t in LABELS:
        n = sum(cm[(t, p)] for p in LABELS)
        note = "   <-- n is tiny; the interval on this spans tens of points" if t == "emergency" else ""
        print(f"recall {t}: {cm[(t, t)]}/{n} = {cm[(t, t)] / n:.0%}{note}")

    n_routine = sum(cm[("routine", p)] for p in LABELS)
    fa = cm[("routine", "same_day")] + cm[("routine", "emergency")]
    print(f"false-alarm rate (true routine flagged up): {fa}/{n_routine} = {fa / n_routine:.0%}")

    # the list that matters to the founder: what would have been buried
    downgrades = [(r, out) for r, out in preds if RANK[out["urgency"]] < RANK[r["urgency"]]]
    print(f"\nDOWNGRADES — true urgent, ranked lower by the system ({len(downgrades)}):")
    for r, out in downgrades:
        print(f"- true={r['urgency']} pred={out['urgency']} score={out['score']}: {r['body'][:160]}")


if __name__ == "__main__":
    main()
