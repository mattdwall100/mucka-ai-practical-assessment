"""Offline fallback predictions: cold artifact + the same escalate-only safety rules.
Same output shape as backfill.py; used when the LLM channel is unavailable.

Run: python ml/predict.py   ->  predictions.csv (no network needed)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

import pandas as pd

from app.service import SCORE_FLOOR, apply_safety_floor
from cold import load_current
from common import load_holdout


def main():
    df = load_holdout().reset_index(drop=True)
    model, cfg = load_current()
    proba = model.predict_proba(df["body"])
    classes = list(model.classes_)
    rows = []
    for i, r in df.iterrows():
        base = classes[int(proba[i].argmax())]
        urgency, hits = apply_safety_floor(r["body"], base)
        score = int(round(100 * (1 - proba[i][classes.index("routine")])))
        if urgency in SCORE_FLOOR:
            score = max(score, SCORE_FLOOR[urgency])
        reason = f"cold model {cfg['version']}" + (f" [safety rule: {', '.join(hits)}]" if hits else "")
        rows.append({"message_id": r["message_id"], "urgency": urgency,
                     "score": score, "reason": reason, "source": "cold_fallback"})
    out = pd.DataFrame(rows)
    # Don't silently replace hot-path output with the weaker fallback — it reads 9/22
    # same_day against the hot channel's 16/22, and the file looks identical either way.
    dest = ROOT / "predictions.csv"
    if dest.exists() and "hot" in pd.read_csv(dest).get("source", pd.Series(dtype=str)).values:
        if "--force" not in sys.argv:
            print(f"{dest.name} came from the hot path; refusing to overwrite. "
                  "Pass --force if that is really what you want.")
            return
    out.to_csv(dest, index=False)
    print(f"wrote predictions.csv ({len(out)} rows) via cold artifact {cfg['version']}")
    print(out["urgency"].value_counts().to_dict())


if __name__ == "__main__":
    main()
