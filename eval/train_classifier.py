"""Train a LightGBM classifier on the public-split labels and report honest
out-of-sample accuracy via GroupKFold (groups = unique (curr, prior) text pair).

Usage:
    python -m eval.train_classifier              # 5-fold CV, no model saved
    python -m eval.train_classifier --save       # 5-fold CV + train on full data + save
    python -m eval.train_classifier --folds 10   # 10-fold instead of 5

The grouping by description pair is deliberate: it tells us how the model
generalises to unseen description pairs, which matches the private-split
inference setting (private split contains some description pairs we have
never trained on).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from app.features import featurize, feature_names  # noqa: E402

DEFAULT_JSON = ROOT / "relevant_priors_public.json"
DEFAULT_MODEL_PATH = ROOT / "app" / "classifier_model.pkl"


def load_pairs(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    truth = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in data["truth"]}

    rows = []
    for c in data["cases"]:
        cd = c["current_study"]["study_description"]
        cdate = c["current_study"].get("study_date")
        for p in c["prior_studies"]:
            key = (c["case_id"], p["study_id"])
            if key not in truth:
                continue
            pd = p["study_description"]
            pdate = p.get("study_date")
            label = truth[key]
            rows.append({
                "case_id": c["case_id"],
                "study_id": p["study_id"],
                "curr_desc": cd,
                "prior_desc": pd,
                "curr_date": cdate,
                "prior_date": pdate,
                "label": label,
            })
    return rows


def build_matrix(rows):
    names = feature_names()
    X = np.zeros((len(rows), len(names)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        fb = featurize(r["curr_desc"], r["prior_desc"], r["curr_date"], r["prior_date"])
        X[i, :] = fb.values
        y[i] = int(r["label"])
    return X, y, names


def train_one(X_train, y_train, params=None):
    p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        p.update(params)
    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(p, train_set, num_boost_round=300)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability threshold for converting to bool")
    args = ap.parse_args()

    print(f"Loading {args.json} ...")
    rows = load_pairs(Path(args.json))
    print(f"  rows: {len(rows)}")
    print(f"  positive rate: {sum(r['label'] for r in rows) / len(rows):.4f}")

    print("Building feature matrix ...")
    t0 = time.perf_counter()
    X, y, names = build_matrix(rows)
    print(f"  X shape: {X.shape}  ({time.perf_counter() - t0:.1f}s)")

    # Group by unique (curr_desc, prior_desc) tuple — same text-pair never spans folds.
    groups = np.array([hash((r["curr_desc"], r["prior_desc"])) for r in rows])
    n_groups = len(set(groups))
    print(f"  unique (curr,prior) text-pair groups: {n_groups}")

    print(f"\n=== {args.folds}-fold GroupKFold CV ===")
    gkf = GroupKFold(n_splits=args.folds)
    oof_proba = np.zeros(len(rows), dtype=np.float32)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        t0 = time.perf_counter()
        model = train_one(X[tr_idx], y[tr_idx])
        proba = model.predict(X[te_idx])
        oof_proba[te_idx] = proba
        pred = (proba >= args.threshold).astype(int)
        acc = (pred == y[te_idx]).mean()
        print(f"  fold {fold}/{args.folds}  train={len(tr_idx):5d} test={len(te_idx):5d}  "
              f"acc={acc:.4f}  ({time.perf_counter() - t0:.1f}s)")

    pred = (oof_proba >= args.threshold).astype(int)
    correct = (pred == y).sum()
    total = len(y)
    acc = correct / total
    confusion = Counter()
    for t, p in zip(y, pred):
        confusion[(bool(t), bool(p))] += 1

    print(f"\n=== Out-of-fold totals (threshold={args.threshold}) ===")
    print(f"accuracy:   {acc:.4f}  ({correct}/{total})")
    print("confusion (true, pred): count")
    for (t, p), n in sorted(confusion.items()):
        print(f"  ({t!s:5}, {p!s:5}) {n}")

    # Threshold sweep — useful to see if 0.5 is optimal
    print("\n=== Threshold sweep on OOF predictions ===")
    print(f"{'thr':>5} {'acc':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        p = (oof_proba >= thr).astype(int)
        a = (p == y).mean()
        tp = ((p == 1) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        tn = ((p == 0) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        print(f"{thr:>5.2f} {a:>7.4f} {tp:>6d} {fp:>6d} {tn:>6d} {fn:>6d}")

    # Top features by gain
    print("\n=== Top 20 features by gain (last fold) ===")
    importance = sorted(zip(names, model.feature_importance(importance_type="gain")),
                        key=lambda x: -x[1])
    for n, imp in importance[:20]:
        print(f"  {n:30s} {imp:>10.1f}")

    if args.save:
        print(f"\nTraining final model on full data and saving ...")
        model = train_one(X, y)
        with open(args.model_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": names, "threshold": args.threshold}, f)
        print(f"  saved to {args.model_path}")


if __name__ == "__main__":
    main()
