---
title: Relevant Priors Classifier
emoji: 🩻
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

## Overview

HTTP API for the **relevant-priors-v1** challenge. For each (current examination,
prior examination) pair the service predicts whether the prior should be shown
to the radiologist.

**Live endpoint:** <https://wothmag07-new-latern-space.hf.space/predict>
(`GET /healthz` returns liveness; `POST /predict` is the contract endpoint).

## API

```http
POST /predict
Content-Type: application/json
```

Request and response shapes match the challenge brief verbatim — see
`app/schemas.py`.

`POST /` is also accepted (some evaluators target the root path). `GET /healthz`
returns liveness + classifier-availability status.

## Approach

Two layered tiers — no external API at inference time.

1. **LightGBM classifier** (`app/classifier_model.py`, `app/features.py`) —
   primary path. Trained offline on the public split with ~120 engineered
   features per pair: region one-hots, modality, contrast, laterality, date
   deltas, description-text features, and the heuristic's own prediction +
   confidence as inputs. Deterministic, ~2 ms/pair, packaged in the image.
2. **Heuristic-only fallback** — fires if `classifier_model.pkl` is missing
   on a deploy or if a feature-schema-drift check at load time refuses the
   model. `app/parser.py` tags each `study_description` with anatomical
   regions + modality + contrast/laterality; `app/heuristic.py` does
   region-set overlap with confidence scoring.
3. **All-False fallback** — handled at the FastAPI layer in `app/main.py`. If
   both tiers above fail or the request exceeds a 300 s wall-clock budget,
   one all-False prediction per prior is returned (76 % baseline). No
   prediction is ever skipped.

Public-split accuracy:

| Predictor | Accuracy | Notes |
| --- | --- | --- |
| Heuristic only | 0.9418 | regex parser + region overlap, no API |
| **LightGBM classifier (5-fold OOF CV)** | **0.9603** | shipped predictor — honest out-of-sample estimate, deterministic with `seed=42` |
| LightGBM (held-out private-split eval) | 0.9361 | reviewer evaluation result on the hidden split |

See `experiments.md` for the full experiment table, ablations, embeddings
comparison, and forward-looking methodology suggestions.

## Pipeline

Two phases that share the same feature-engineering code in `app/features.py`,
which guarantees train/inference parity (no skew).

### Training (offline, run once via `python -m eval.train_classifier --save`)

```text
relevant_priors_public.json (27,614 labeled pairs)
        │
        ▼
load_pairs()  →  list of {curr_desc, prior_desc, curr_date, prior_date, label}
        │
        ▼
For each pair: featurize(...)
       │   parse_description(curr_desc) → StudyTags(regions, modality, ...)
       │   parse_description(prior_desc) → StudyTags(...)
       │   classify_pair(curr_tags, prior_tags) → HeuristicResult (used as features)
       ▼
   ~120 numeric features per pair
        │
        ▼
X (27614 × 121),  y (27614,)
        │
        ▼
GroupKFold (groups = unique curr+prior text pair) → honest 5-fold CV (0.9603 OOF)
        │
        ▼
lgb.train(full_data) → app/classifier_model.pkl    (1 MB pickle, packaged in image)
```

### Inference (online, every `POST /predict` request)

```text
POST /predict {cases: [...]}
        │
        ▼
app/main.py::_handle      ← wraps in 300 s wall-clock budget (asyncio.wait_for)
        │
        ▼
app/classifier.py::predict_cases_async
        │
        ▼
┌──────────────────────────────────┐
│ Tier 1: LightGBM                 │ ← fires if classifier_model.pkl is present
│   _predict_via_classifier        │   AND its feature schema matches features.py
│   → featurize each pair          │   (the production path)
│   → model.predict(X)             │
│   → bool[i] = proba[i] >= 0.5    │
└──────────┬───────────────────────┘
           │ model missing or schema drift? ↓
┌──────────▼───────────────────────┐
│ Tier 2: heuristic only           │ ← defensive fallback
│   parser → classify_pair()       │   (no API, fully self-contained)
│   → bool from HeuristicResult    │
└──────────┬───────────────────────┘
           │ tier 2 raises or budget expires? ↓
┌──────────▼───────────────────────┐
│ Tier 3: all-False                │ ← ultimate safety net at FastAPI layer
│   one False per prior            │   (returns 76 % baseline, never skips)
└──────────────────────────────────┘
        │
        ▼
HTTP 200 {predictions: [{case_id, study_id, predicted_is_relevant}, ...]}
```

### File responsibilities

| File | Responsibility | Used in training | Used in inference |
| --- | --- | :---: | :---: |
| [`app/parser.py`](app/parser.py) | text → `StudyTags(regions, modality, contrast, laterality)` | ✓ | ✓ |
| [`app/heuristic.py`](app/heuristic.py) | `(StudyTags, StudyTags) → HeuristicResult` (predicted, confidence, reason) | ✓ | ✓ |
| [`app/features.py`](app/features.py) | `(curr, prior, dates, tags) → 121-dim numeric vector` | ✓ | ✓ |
| [`eval/train_classifier.py`](eval/train_classifier.py) | Build X/y, GroupKFold CV, train final, save pickle | ✓ | — |
| `app/classifier_model.pkl` | Pickled `{model, feature_names, threshold}` | output | input |
| [`app/classifier_model.py`](app/classifier_model.py) | Lazy-load pickle; `predict_batch(pairs) → list[bool]` | — | ✓ |
| [`app/classifier.py`](app/classifier.py) | Tier-routing: classifier → heuristic | — | ✓ |
| [`app/main.py`](app/main.py) | FastAPI endpoint + 300 s wall-clock budget | — | ✓ |

### What "the model" actually is

A LightGBM gradient-boosted tree (300 boosted trees, ~1 MB pickled), trained
on engineered features. It is **not** a Hugging Face Hub model — HF Spaces is
just the Docker host. The pickle ships inside the image at
`app/classifier_model.pkl` and is loaded lazily on the first request. The
inference path has no external API dependency.

Top features by gain (training the final model on the full public split):

| Rank | Feature | What it captures |
| --- | --- | --- |
| 1 | `heur_pred` | The heuristic's prediction itself — dominant input |
| 2 | `expanded_overlap` | Region overlap with coverage expansions (e.g. `abd_pel ⊇ {abdomen,pelvis}`) |
| 3 | `date_delta_days` | Days between current and prior study (signal the heuristic ignored) |
| 4 | `heur_conf` | Heuristic confidence as a second-order signal |
| 5 | `lateral_mismatch` | Left↔right detection (parsed but never used by heuristic) |
| 6 | `direct_overlap` | Literal region-set intersection |
| 7 | `shared_tokens` | Token-level text similarity |
| 8 | `desc_len_ratio` | Length ratio of the two descriptions |

The classifier essentially **augments** the heuristic with date deltas,
laterality, and text-level similarity — three signals the heuristic doesn't
look at. That's where the +1.85 pp jump from heuristic (0.9418) to classifier
OOF (0.9603) comes from.

## Local development

```bash
# uv-based setup (recommended)
uv venv --python 3.11
source .venv/Scripts/activate          # Windows; on macOS/Linux: source .venv/bin/activate
uv pip install -r requirements.txt

pytest tests/ -q                                 # contract + parser unit tests
python -m eval.run_eval                          # scores the classifier (~65 s)
python -m eval.run_eval --predictor heuristic    # heuristic-only baseline
python -m eval.train_classifier                  # 5-fold CV (deterministic with seed=42)
python -m eval.train_classifier --save           # retrain on full data, save app/classifier_model.pkl

uvicorn app.main:app --reload                    # dev server on :8000
```

## Deployment (Hugging Face Spaces)

1. Create a new Space → SDK = Docker.
2. Push this repo. The Dockerfile builds a Python 3.11 image, installs
   `requirements.txt`, and copies `app/` (including the trained
   `classifier_model.pkl`).
3. The `app_port: 7860` in the metadata tells HF where to send traffic.
4. No secrets required — the inference path is fully self-contained.

## Environment variables

| Var | Default | Notes |
| --- | --- | --- |
| `CLASSIFIER_MODEL_PATH` | `app/classifier_model.pkl` | Override path to the trained LightGBM model. |
| `CLASSIFIER_THRESHOLD` | `0.5` (or value baked into the pkl) | Probability threshold for the classifier. |
| `REQUEST_TIMEOUT_S` | `300` | Hard wall-clock budget per `/predict` request. On expiry, in-flight tasks are cancelled and the all-False fallback fires (well inside the evaluator's 360 s timeout). |
| `LOG_LEVEL` | `INFO` | Standard Python logging level. |
