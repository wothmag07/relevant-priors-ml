"""Compare Option 2 (engineered features only) vs Option 3 (engineered + sentence
embeddings) on identical 5-fold GroupKFold CV.

We try three variants:
- Option 2: current feature matrix (~50 features, the deployed model)
- Option 3a: + cosine similarity between curr/prior MiniLM embeddings (1 extra feature)
- Option 3b: + cosine + element-wise diff/sum summaries (~10 extra features)

Embeddings are pre-computed once over unique descriptions and reused across pairs,
so total compute is small (~1000 unique descriptions × MiniLM CPU encode = a few seconds).

Usage:
    python -m eval.compare_embeddings
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from app.features import feature_names, featurize  # noqa: E402

DEFAULT_JSON = ROOT / "relevant_priors_public.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
            rows.append({
                "case_id": c["case_id"], "study_id": p["study_id"],
                "curr_desc": cd, "prior_desc": p["study_description"],
                "curr_date": cdate, "prior_date": p.get("study_date"),
                "label": truth[key],
            })
    return rows


def precompute_embeddings(rows):
    """Embed each unique description once; return dict desc -> np.ndarray."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    unique = sorted({r["curr_desc"] for r in rows} | {r["prior_desc"] for r in rows})
    print(f"  encoding {len(unique)} unique descriptions ...")
    t0 = time.perf_counter()
    embs = model.encode(unique, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    print(f"  done in {time.perf_counter() - t0:.1f}s — embedding dim={embs.shape[1]}")
    return dict(zip(unique, embs, strict=True))


def build_base_matrix(rows):
    names = feature_names()
    X = np.zeros((len(rows), len(names)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        fb = featurize(r["curr_desc"], r["prior_desc"], r["curr_date"], r["prior_date"])
        X[i, :] = fb.values
        y[i] = int(r["label"])
    return X, y, names


def build_embedding_features(rows, emb_lookup, mode: str):
    """mode: 'cos' = single cosine; 'cos_summary' = cosine + summary stats over diff/sum."""
    n = len(rows)
    feats = []
    feat_names = []
    cos = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(rows):
        ce = emb_lookup[r["curr_desc"]]
        pe = emb_lookup[r["prior_desc"]]
        cos[i] = float(np.dot(ce, pe))  # already normalized
    feats.append(cos)
    feat_names.append("emb_cos_sim")

    if mode == "cos_summary":
        diffs_l1 = np.zeros(n, dtype=np.float32)
        diffs_l2 = np.zeros(n, dtype=np.float32)
        sums_max = np.zeros(n, dtype=np.float32)
        sums_min = np.zeros(n, dtype=np.float32)
        prods_mean = np.zeros(n, dtype=np.float32)
        prods_std = np.zeros(n, dtype=np.float32)
        diffs_max = np.zeros(n, dtype=np.float32)
        diffs_std = np.zeros(n, dtype=np.float32)
        for i, r in enumerate(rows):
            ce = emb_lookup[r["curr_desc"]]
            pe = emb_lookup[r["prior_desc"]]
            d = ce - pe
            s = ce + pe
            p = ce * pe
            diffs_l1[i] = np.abs(d).sum()
            diffs_l2[i] = float(np.linalg.norm(d))
            sums_max[i] = s.max()
            sums_min[i] = s.min()
            prods_mean[i] = p.mean()
            prods_std[i] = p.std()
            diffs_max[i] = np.abs(d).max()
            diffs_std[i] = d.std()
        feats.extend([diffs_l1, diffs_l2, sums_max, sums_min, prods_mean, prods_std, diffs_max, diffs_std])
        feat_names.extend(["emb_diff_l1", "emb_diff_l2", "emb_sum_max", "emb_sum_min",
                           "emb_prod_mean", "emb_prod_std", "emb_diff_max", "emb_diff_std"])
    return np.column_stack(feats).astype(np.float32), feat_names


def cv_score(X, y, groups, n_folds=5, label=""):
    gkf = GroupKFold(n_splits=n_folds)
    oof_proba = np.zeros(len(y), dtype=np.float32)
    fold_accs = []
    for _fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
            "verbose": -1, "n_jobs": -1,
        }
        model = lgb.train(params, lgb.Dataset(X[tr_idx], label=y[tr_idx]), num_boost_round=300)
        proba = np.asarray(model.predict(X[te_idx]))
        oof_proba[te_idx] = proba
        acc = ((proba >= 0.5).astype(int) == y[te_idx]).mean()
        fold_accs.append(acc)
    pred = (oof_proba >= 0.5).astype(int)
    overall = (pred == y).mean()
    confusion: Counter = Counter()
    for t, p in zip(y, pred, strict=True):
        confusion[(bool(t), bool(p))] += 1
    print(f"\n=== {label} ===")
    print(f"  features: {X.shape[1]}")
    print(f"  fold accs: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"  OOF acc:   {overall:.4f}  ({(pred == y).sum()}/{len(y)})")
    print("  confusion:")
    for (t, p), n in sorted(confusion.items()):
        print(f"    ({t!s:5}, {p!s:5}) {n}")
    return overall, model


def main():
    rows = load_pairs(DEFAULT_JSON)
    print(f"loaded {len(rows)} rows; positive rate = {sum(r['label'] for r in rows)/len(rows):.4f}")

    emb_lookup = precompute_embeddings(rows)

    print("\nBuilding base feature matrix (Option 2) ...")
    Xb, y, base_names = build_base_matrix(rows)
    groups = np.array([hash((r["curr_desc"], r["prior_desc"])) for r in rows])

    print("\nBuilding +cosine feature ...")
    Xe1, e1_names = build_embedding_features(rows, emb_lookup, mode="cos")
    X3a = np.hstack([Xb, Xe1])

    print("\nBuilding +cos_summary feature set ...")
    Xe2, e2_names = build_embedding_features(rows, emb_lookup, mode="cos_summary")
    X3b = np.hstack([Xb, Xe2])

    a2, m2 = cv_score(Xb, y, groups, label="Option 2  (engineered only)")
    a3a, m3a = cv_score(X3a, y, groups, label="Option 3a (+ cosine sim)")
    a3b, m3b = cv_score(X3b, y, groups, label="Option 3b (+ cosine + summaries)")

    print("\n=== Comparison ===")
    print(f"  Option 2:   {a2:.4f}")
    print(f"  Option 3a:  {a3a:.4f}  ({(a3a - a2) * 100:+.2f} pp)")
    print(f"  Option 3b:  {a3b:.4f}  ({(a3b - a2) * 100:+.2f} pp)")

    # Top features in 3b for diagnosis
    print("\nTop 15 features in Option 3b (last fold):")
    all_names = base_names + e2_names
    importance = sorted(zip(all_names, m3b.feature_importance(importance_type="gain"), strict=True),
                        key=lambda x: -x[1])
    for n, imp in importance[:15]:
        marker = "  <- embedding" if n.startswith("emb_") else ""
        print(f"  {n:30s} {imp:>10.1f}{marker}")


if __name__ == "__main__":
    main()
