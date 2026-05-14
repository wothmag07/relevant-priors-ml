# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3.11
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Worked example: how the classifier sees a single (current, prior) pair
#
# This walkthrough takes one (current study, prior study) pair end-to-end
# through the pipeline so you can see *why* the classifier predicts what it
# does — not just what it predicts.
#
# Run as a script:
#
#     python docs/walkthrough.py
#
# Or convert to a notebook:
#
#     jupytext --to ipynb docs/walkthrough.py
#     jupyter notebook docs/walkthrough.ipynb

# %%
from __future__ import annotations
import sys
from pathlib import Path

# Make the repo root importable when this file is run from docs/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows default cp1252 stdout can't print unicode; reconfigure for safety.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import app.classifier_model as cm  # noqa: E402
from app.features import featurize, feature_names  # noqa: E402
from app.heuristic import classify_pair  # noqa: E402
from app.parser import parse_description  # noqa: E402

# %% [markdown]
# ## Step 1 — pick a pair
#
# A real-looking pair where the heuristic is uncertain: a current chest CT
# and a prior abdomen/pelvis CT. They share the "ct" modality and partially
# overlap on coverage (chest doesn't touch abdomen, but `chest_abd_pel` would).
# The classifier has to decide based on more than just the heuristic.

# %%
CURR_DESC = "CT CHEST WITH CONTRAST"
CURR_DATE = "2026-03-08"

PRIOR_DESC = "CT ABDOMEN AND PELVIS WITH CONTRAST"
PRIOR_DATE = "2025-08-15"

# %% [markdown]
# ## Step 2 — what does the parser extract?

# %%
curr_tags = parse_description(CURR_DESC)
prior_tags = parse_description(PRIOR_DESC)
print("Current tags:", curr_tags)
print("Prior tags:  ", prior_tags)

# %% [markdown]
# ## Step 3 — what does the heuristic say?

# %%
heur = classify_pair(curr_tags, prior_tags)
print(f"Heuristic predicted={heur.predicted}  confidence={heur.confidence:.2f}")
print(f"Reason: {heur.reason}")

# %% [markdown]
# ## Step 4 — featurize for the classifier
#
# `featurize()` produces a ~120-dim numeric vector that includes the heuristic's
# own output as features (`heur_pred`, `heur_conf`) plus region overlap, date
# delta, text similarity, modality one-hots, etc.

# %%
fb = featurize(CURR_DESC, PRIOR_DESC, CURR_DATE, PRIOR_DATE)
names = feature_names()
values = np.asarray(fb.values, dtype=np.float32)
print(f"Feature vector length: {values.shape[0]}")

# Show the non-zero features so we see what's "on" for this pair
nz = [(n, float(v)) for n, v in zip(names, values, strict=True) if v != 0.0]
print(f"\nNon-zero features for this pair ({len(nz)} / {len(names)}):")
for n, v in nz:
    print(f"  {n:30s}  {v:8.3f}")

# %% [markdown]
# ## Step 5 — classifier prediction + per-feature SHAP-style contributions
#
# LightGBM exposes `predict(pred_contrib=True)`, which returns the log-odds
# contribution of every feature for this single prediction. That tells us
# *which features pushed the decision in which direction* — the explainability
# story.

# %%
assert cm._load_once(), "Train the model first: python -m eval.train_classifier --save"
assert cm._model is not None and cm._feature_names is not None

X = values.reshape(1, -1)
proba = float(cm._model.predict(X)[0])
print(f"Classifier P(relevant) = {proba:.4f}")
print(f"Classifier prediction  = {bool(proba >= 0.5)}")

# pred_contrib returns shape (1, n_features + 1) — the trailing column is the
# baseline log-odds (the "expected value" before any feature contributes).
contribs = cm._model.predict(X, pred_contrib=True)[0]
baseline_logit = float(contribs[-1])
feat_contribs = contribs[:-1]

print(f"\nBaseline log-odds (before any feature):  {baseline_logit:+.3f}")
print(f"Final log-odds:                          {baseline_logit + feat_contribs.sum():+.3f}")
print(f"Final probability (sigmoid):             {1 / (1 + np.exp(-(baseline_logit + feat_contribs.sum()))):.4f}")

# %% [markdown]
# ## Step 6 — top contributing features for this prediction

# %%
ranked = sorted(
    zip(cm._feature_names, feat_contribs, strict=True),
    key=lambda x: -abs(x[1]),
)
print("Top 12 features for this prediction (sorted by |contribution|):")
print(f"  {'feature':30s}  {'logit Δ':>10s}  direction")
for n, c in ranked[:12]:
    direction = "→ relevant" if c > 0 else "→ NOT relevant"
    print(f"  {n:30s}  {c:+10.3f}  {direction}")
