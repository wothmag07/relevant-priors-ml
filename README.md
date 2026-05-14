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
returns liveness + LLM-enabled status.

## Approach

Three layered tiers. The first to succeed wins.

1. **LightGBM classifier** (`app/classifier_model.py`, `app/features.py`) —
   primary path. Trained offline on the public split with ~120 engineered
   features per pair: region one-hots, modality, contrast, laterality, date
   deltas, description-text features, and the heuristic's own prediction +
   confidence as inputs. Deterministic, ~2 ms/pair, no API dependency.
2. **Heuristic + LLM hybrid** — fallback if the LightGBM model file is missing
   on a deploy. `app/parser.py` tags each `study_description` with anatomical
   regions + modality + contrast/laterality; `app/heuristic.py` does region-set
   overlap; pairs with confidence `< 0.85` are batched to `gpt-4o-mini`.
3. **All-False fallback** — handled at the FastAPI layer in `app/main.py`. If
   both tiers above fail or the request exceeds a 300 s wall-clock budget,
   one all-False prediction per prior is returned (76 % baseline). No
   prediction is ever skipped.

Public-split accuracy:

| Predictor | Accuracy | Notes |
| --- | --- | --- |
| Heuristic only | 0.9408 | regex parser + region overlap, no API |
| Hybrid (heuristic + `gpt-4o-mini`) | 0.9491 | previous architecture |
| **LightGBM classifier (5-fold OOF CV)** | **0.9603** | shipped predictor — honest out-of-sample estimate |

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
┌──────────────────────────┐
│ Tier 1: LightGBM         │ ← fires if app/classifier_model.pkl exists
│   _predict_via_classifier│   (the production path)
│   → featurize each pair  │
│   → model.predict(X)     │
│   → bool[i] = p[i] >= 0.5│
└──────────┬───────────────┘
           │ model missing? ↓
┌──────────▼───────────────┐
│ Tier 2: heuristic + LLM  │ ← defensive fallback
│   parser → heuristic     │
│   conf < 0.85 → gpt-4o-mini (one batched call per case)
│   results cached on disk │
└──────────┬───────────────┘
           │ tier 2 raises or budget expires? ↓
┌──────────▼───────────────┐
│ Tier 3: all-False        │ ← ultimate safety net at FastAPI layer
│   one False per prior    │   (returns 76 % baseline, never skips)
└──────────────────────────┘
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
| [`app/classifier.py`](app/classifier.py) | Tier-routing across the three layers above | — | ✓ |
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
look at. That's where the +1.95 pp jump from heuristic (0.9408) to classifier
OOF (0.9603) comes from.

## Local development

```bash
# uv-based setup (recommended)
uv venv --python 3.11
source .venv/Scripts/activate          # Windows; on macOS/Linux: source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env                   # OPENAI_API_KEY only needed for hybrid fallback path

pytest tests/ -q                       # contract tests (no API key needed)
python -m eval.run_eval --predictor classifier   # primary path, ~65 s on full public split
python -m eval.run_eval --predictor heuristic    # heuristic-only baseline
LLM_ENABLED=1 python -m eval.run_eval --predictor hybrid   # heuristic + LLM fallback path
python -m eval.train_classifier --save           # retrain classifier and save to app/classifier_model.pkl

uvicorn app.main:app --reload                    # dev server on :8000
```

## Deployment (Hugging Face Spaces)

1. Create a new Space → SDK = Docker.
2. Push this repo. The Dockerfile builds a Python 3.11 image, installs
   `requirements.txt`, and copies `app/` (including the trained
   `classifier_model.pkl`) and any pre-warmed `.cache/` directory.
3. Set `OPENAI_API_KEY` as a Space secret only if you want the LLM fallback
   path active. The classifier tier is the primary path and needs no secret.
4. The `app_port: 7860` in the metadata tells HF where to send traffic.

## Environment variables

| Var | Default | Notes |
| --- | --- | --- |
| `CLASSIFIER_MODEL_PATH` | `app/classifier_model.pkl` | Override path to the trained LightGBM model. |
| `CLASSIFIER_THRESHOLD` | `0.5` (or value baked into the pkl) | Probability threshold for the classifier. |
| `OPENAI_API_KEY` | (unset) | Only needed for the LLM fallback tier. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any chat model that supports `response_format=json_object`. |
| `LLM_ENABLED` | `1` | Set to `0` to force the heuristic to run alone in the fallback tier. |
| `LLM_MAX_CONCURRENCY` | `8` | Cap concurrent OpenAI calls per request (fallback tier). |
| `LLM_TIMEOUT_S` | `60` | Per-call timeout for OpenAI requests. |
| `REQUEST_TIMEOUT_S` | `300` | Hard wall-clock budget per `/predict` request. On expiry, in-flight tasks are cancelled and the all-False fallback fires. |
| `CACHE_PATH` | `.cache/llm_cache.json` | Disk-backed LLM cache (fallback tier). |
