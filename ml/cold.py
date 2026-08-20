"""Cold path: offline retraining service.

Reads the processed training set (stand-in for the production DB), de-conflicts,
fits TF-IDF + logistic regression, calibrates the alert threshold to the office's
alert budget, and emits a versioned artifact under data/config/vN/.

Promotion = data/config/current.txt naming a version. Rollback = editing it back.
The cold path never touches live routing directly.

Run: python ml/cold.py    (or POST /retrain on the API)
"""
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from common import DATA, PROCESSED, URGENCY_ORDER, build_vectorizer, load_messages

CONFIG = DATA / "config"
DAILY_VOLUME = 90        # midpoint of the stated 60-120 inbound/day
MAX_ALERTS_PER_DAY = 10  # if everything is urgent nothing is: cap flags, report recall AT the cap


def retrain(train_csv=None):
    train_csv = train_csv or PROCESSED / "train.csv"
    # ponytail: CSV stands in for the DB; real impl = SELECT from corpus + overrides
    df = pd.read_csv(train_csv)

    # de-conflict: identical normalised body, different labels -> keep the more urgent
    n_conflicts = int(df.groupby("core_hash")["label"].nunique().gt(1).sum())
    rank = df["label"].map(URGENCY_ORDER.index)
    df = df.loc[rank.sort_values(ascending=False).index].drop_duplicates("core_hash")

    model = Pipeline([("vec", build_vectorizer()),
                      ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])

    # honest per-row scores via out-of-fold CV, then fit on everything
    proba = cross_val_predict(model, df["body"], df["label"], cv=5, method="predict_proba")
    classes = list(np.unique(df["label"]))
    p_urgent = 1 - proba[:, classes.index("routine")]
    model.fit(df["body"], df["label"])

    # threshold = lowest flag bar that still fits the alert budget, on the real traffic mix
    p_all = 1 - model.predict_proba(load_messages()["body"])[:, classes.index("routine")]
    grid = np.linspace(0.05, 0.95, 181)
    threshold = float(next(t for t in grid
                           if (p_all >= t).mean() * DAILY_VOLUME <= MAX_ALERTS_PER_DAY))
    est_alerts = float((p_all >= threshold).mean() * DAILY_VOLUME)

    gold = df["label_source"] == "gold_train"   # recall reported on real labels only
    urgent = (df["label"] != "routine") & gold
    em = (df["label"] == "emergency") & gold

    CONFIG.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name[1:]) for p in CONFIG.glob("v*") if p.is_dir()], default=0) + 1
    vdir = CONFIG / f"v{n}"
    vdir.mkdir()
    with open(vdir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    summary = {
        "version": f"v{n}",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "conflicts_resolved_upward": n_conflicts,
        "threshold": round(threshold, 3),
        "est_alerts_per_day": round(est_alerts, 1),
        "urgent_recall_at_budget": f"{int(((p_urgent >= threshold) & urgent).sum())}/{int(urgent.sum())}",
        "emergency_recall_at_budget":
            f"{int(((p_urgent >= threshold) & em).sum())}/{int(em.sum())} (n tiny; interval spans tens of points)",
        "classes": classes,
    }
    (vdir / "config.json").write_text(json.dumps(summary, indent=2))
    (CONFIG / "current.txt").write_text(f"v{n}")  # promotion; roll back by editing this file
    return summary


def load_current():
    v = (CONFIG / "current.txt").read_text().strip()
    with open(CONFIG / v / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return model, json.loads((CONFIG / v / "config.json").read_text())


if __name__ == "__main__":
    print(json.dumps(retrain(), indent=2))
