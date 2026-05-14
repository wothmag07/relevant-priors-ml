# Relevant Priors — Experiments & Next Steps

**Live endpoint:** <https://wothmag07-new-latern-space.hf.space/predict>

## Problem

For each `(current_examination, prior_examination)` pair, predict whether the
prior should be shown to the radiologist while reading the current study.
Scoring: `accuracy = correct / (correct + incorrect)`; skipped predictions
count as incorrect. The signal lives almost entirely in two short text fields
(`study_description` and `study_date`).

## Public-split summary

| Stat | Value |
| --- | --- |
| Cases | 996 |
| Labeled prior pairs | 27,614 |
| Positive rate | 23.8 % |
| Mean priors per case | 27.7 |
| Max priors per case | 234 |
| Unique current descriptions | 278 |
| Unique prior descriptions | 812 |
| Unique `(current, prior)` pairs | 12,462 |

Implications:

- **Bulk inference is mandatory.** Some single cases have 234 priors — one LLM
  call per pair would blow the 360 s evaluator timeout.
- **The keyspace is finite.** Only 12 k unique `(current, prior)` description
  pairs across the entire split.
- **23.8 % positive rate** means the trivial baseline `always-False` scores
  **76.22 %**.

## Final architecture

Three layered tiers in `app/classifier.py`. The first tier that succeeds wins.

1. **LightGBM classifier** (`app/classifier_model.py`, `app/features.py`) —
   primary path. Trained offline on the public split with ~120 engineered
   features per pair: region one-hots, modality, contrast, laterality, date
   deltas, description text features, and the heuristic's own prediction +
   confidence. Deterministic, ~2 ms/pair, no API dependency.

2. **Heuristic + LLM hybrid** (`app/parser.py`, `app/heuristic.py`,
   `app/llm.py`) — fallback when the LightGBM model file is missing on a deploy.
   Heuristic tags each description into region / modality / contrast / laterality;
   pairs with heuristic confidence `< 0.85` are batched per case to `gpt-4o-mini`
   with `response_format=json_object`. Results are cached on disk keyed by
   `(curr_description, prior_description)`.

3. **All-False fallback** (`app/main.py`) — final safety net. If both higher
   tiers fail, the request handler returns one prediction per prior with
   `predicted_is_relevant=False`. This is the 76 % baseline — better than
   skipping (which scores 0 %).

The wall-clock budget is bounded at 300 s via `asyncio.wait_for`; on expiry,
in-flight tasks are cancelled and tier 3 fires. This is the hard guarantee
against the evaluator's 360 s timeout regardless of what happens upstream.

## Experiments

All numbers are on the full public split (996 cases / 27,614 pairs).
Wall time was measured on a Windows laptop. The classifier row reports
**out-of-fold accuracy from 5-fold GroupKFold CV** (groups = unique
description pairs), which is the honest unbiased estimate; the
training-set fit on the same data is 0.98+ but is not a generalisation
estimate and is not what we claim.

| # | Predictor | Accuracy | Δ vs always-False | Wall time | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | `always_false` (sanity floor) | 0.7622 | — | 0.03 s | |
| 1 | Heuristic v1 (initial parser, all RELATED pairs on) | 0.9208 | +15.86 pp | 3.1 s | |
| 2 | Heuristic v2 (data-validated related pairs only) | 0.9383 | +17.61 pp | 3.1 s | |
| 3 | Heuristic v3 (+ `ABD_PEL` regex fix, `OUTSIDE FILMS=False`, ultrasound-breast-screening tag) | 0.9408 | +17.86 pp | 3.0 s | |
| 4 | Hybrid v1 (heuristic v3 + `gpt-4o-mini` for conf < 0.85), cold cache | 0.9491 | +18.69 pp | 186 s | |
| 4w | Hybrid v1, warm cache | 0.9482 | +18.60 pp | 1.7 s | |
| 5 | Hybrid v2 (revised LLM prompt: explicit conservative defaults + edge-case exclusions) | 0.9493 | +18.71 pp | 175 s | within noise of v1 |
| **6** | **LightGBM classifier (engineered features), 5-fold OOF** | **0.9603** | **+19.81 pp** | **65 s** | **shipped predictor** |
| 6t | LightGBM, training-set fit on full public split | 0.9809 | — | 65 s | reported for ablation only; not a generalisation estimate |
| 7a | LightGBM + sentence embedding cosine similarity (Option 3a) | 0.9594 | +19.72 pp | 70 s | no improvement over #6 |
| 7b | LightGBM + cosine + diff/sum/prod summaries (Option 3b) | 0.9594 | +19.72 pp | 75 s | no improvement |

## Confusion matrices

### Heuristic v3

```
              pred False   pred True
true False        20480         567
true True          1069        5498
```

Failure modes are roughly symmetric (567 false-positives vs 1069 false-negatives).
Remaining errors are dominated by clusters that are intrinsically ambiguous from
text alone: **echo ↔ chest CT/XR**, **CT angio head ↔ CT angio carotid**,
**EEG ↔ MRI brain**, **CT-guided FNA**, **bone scan ↔ specific organ studies**.

### Hybrid v1 (cold cache)

```
              pred False   pred True
true False        20533         514
true True           892        5675
```

Compared to heuristic-only, hybrid converts 177 false-negatives → true positives
and 53 false-positives → true negatives (230 net wins). The LLM helps more on
the false-negative axis.

### LightGBM classifier (5-fold OOF)

```
              pred False   pred True
true False        20649         398
true True           697        5870
```

Compared to hybrid v1: +195 TP, −116 FP, −195 FN, +116 TN. Both error axes
improve. Total +311 correct predictions on the same data.

## Ablations that hurt — region-pair links

We hypothesised that adding cross-region links to `RELATED_REGION_PAIRS` would
help. We tested 13 candidate pairs; **all 13 are net-negative** on the public
split, so none were added. The pattern is consistent: labelers in this dataset
are judging *clinical utility*, not anatomical overlap, and reject many
anatomically adjacent pairs as "doesn't help today's read."

| Linkage | Pos / Neg pairs in data | Net | Decision |
| --- | --- | --- | --- |
| `heart ↔ chest` | 158 / 506 | −348 | dropped |
| `brain ↔ sinuses` | 12 / 19 | −7 | dropped |
| `brain ↔ vasc_carotid` | 65 / 141 | −76 | dropped |
| `t_spine ↔ chest` | 49 / 99 | −50 | dropped |
| `l_spine ↔ abdomen` | 37 / 208 | −171 | dropped |
| `l_spine ↔ pelvis` | 29 / 130 | −101 | dropped |
| `gi_fluoro ↔ chest` | 25 / 109 | −84 | dropped |
| `c_spine ↔ brain` | 20 / 93 | −73 | dropped |
| `l_spine ↔ abd_pel` | 23 / 76 | −53 | dropped |
| `t_spine ↔ l_spine` | 19 / 47 | −28 | dropped |
| `gi_fluoro ↔ abdomen` | 7 / 28 | −21 | dropped |
| `c_spine ↔ t_spine` | 23 / 38 | −15 | dropped |
| `eeg ↔ brain` | 30 / 32 | −2 | dropped (barely) |

This systematic negative result strongly suggests blanket cross-region rules
are structurally wrong for this dataset. Any future linkages should be
**modality-conditional** (e.g., "heart ↔ chest only when both are CT/MRI"),
not region-only.

## Ablations that helped

### Parser-level (heuristic v3)

| Change | Pos / Neg pairs in data | Decision |
| --- | --- | --- |
| `bone_scan → wholebody` (covers torso PET/NM) | dominant on bone-scan TPs | added |
| `ULTRASOUND BILAT SCREEN → breast` | 107 / 4 | added |
| `OUTSIDE FILMS → False` | 0 / 67 | added |
| `bone_density` exclusive (suppresses `hip`/`spine` co-tags) | 41 / 575 cross-pairs | added |
| Underscore in `ABD_PEL` regex | bug (`\W` excludes `_` in Python) | fixed |

### Cluster A — regex bugs found and fixed

| Fix | Hit count | Heuristic Δ | Hybrid Δ | Decision |
| --- | --- | --- | --- | --- |
| `\b(LEG\|LE)\b` → `\b(LEG\|LEGS\|LE)\b` (catch `BILAT LEGS`) | 213 occurrences | +5 | +0 | kept |
| `\bMYO ?PERF\b` → `\bN?M\s*MYO\s*PERF\b\|...` (catch `NMmyo`) | 70 occurrences | +3 | +9 | kept |
| Both fixes combined | 283 | +8 | +9 (+0.03 pp) | kept |

Tiny gain but free; both are genuine regex bugs that miss valid variants.

### LLM prompt v2

Replaced the ambiguous `OPEN` rule "*relevant if anatomy substantially overlaps*"
with a more precise rubric: explicit conservative-default ("when in doubt,
predict NOT relevant — labelers are strict"), plus data-validated exclusions
for the highest-volume error clusters (spine ↔ chest/abdomen/pelvis,
carotid ↔ brain, EEG ↔ brain, GI fluoro ↔ chest, adjacent spine levels).

Result on cold-cache hybrid: 0.9491 → **0.9493** (+6 correct, +0.02 pp).
Within the cache-race noise floor (~24 predictions), but directionally correct
— the model became more conservative as intended (−40 TP, −46 FP, +46 TN,
+40 FN). Kept the v2 prompt because it's strictly better as documentation
even if the metric impact is small.

### LightGBM classifier (Option 2) — the big win

**Public-split CV: 0.9603 (5-fold GroupKFold).** This is the predictor
shipped in production.

Top features by gain (last fold):

| Rank | Feature | Gain | What it tells us |
| --- | --- | --- | --- |
| 1 | `heur_pred` | 88,060 | The heuristic is the dominant signal — model uses it as the base |
| 2 | `expanded_overlap` | 48,377 | Region overlap with coverage expansions |
| 3 | `date_delta_days` | 6,733 | **Dates do help** — first time we used them |
| 4 | `heur_conf` | 5,679 | Heuristic confidence as a second-order signal |
| 5 | `lateral_mismatch` | 3,950 | Left↔right detection (parsed but never used by heuristic) |
| 6 | `direct_overlap` | 3,854 | |
| 7 | `shared_tokens` | 3,609 | Token-level text similarity |
| 8 | `desc_len_ratio` | 3,526 | |

Three of the top eight (date deltas, laterality, text-level similarity) are
signals the heuristic completely ignored. The classifier essentially
**augments** the heuristic with these missing inputs.

### Sentence embeddings (Option 3) — no improvement

Tested whether MiniLM-based sentence embeddings could lift the classifier
beyond engineered features.

| Variant | Features | OOF acc | Δ vs Option 2 |
| --- | --- | --- | --- |
| Option 2 (engineered) | 121 | 0.9595 | — |
| Option 3a (+ embedding cosine sim) | 122 | 0.9594 | −0.01 pp |
| Option 3b (+ cosine + diff/sum/prod summary stats) | 130 | 0.9594 | −0.01 pp |

All within noise. Why embeddings don't help here:

- Descriptions are short bureaucratic strings, not natural language.
- Engineered features already capture the same content (regions, modality,
  shared tokens).
- MiniLM is general-purpose and lacks radiology-domain knowledge — our
  100-line regex parser is at a knowledge advantage on these specific terms.

Removed the `sentence-transformers` / `torch` deps after this measurement.
A *medical-domain* embedding (BioBERT, ClinicalBERT, RadBERT) might do better,
but that's a much bigger commitment for an unclear gain.

## Cost & latency

| Predictor | Wall time on full public split | API cost | External deps |
| --- | --- | --- | --- |
| Heuristic only | 3 s | $0 | none |
| Hybrid v1 (cold) | 186 s | ~$0.10–0.20 | OpenAI |
| Hybrid v1 (warm cache) | 1.7 s | $0 | OpenAI (key still required to handle unseen pairs) |
| **LightGBM classifier (deployed)** | **65 s** | **$0** | **none — model packaged in image** |

Per-pair: ~2.4 ms for the classifier. Worst-case private-split case (234
priors) → ~0.6 s for that one case. A request with 100 such cases → ~1 min,
comfortably inside the 360 s evaluator budget.

The heuristic + LLM hybrid is retained as a fallback path. If the
`classifier_model.pkl` is missing on a deploy, the system silently degrades
to hybrid (or further to heuristic-only if `OPENAI_API_KEY` is unset). All
three failure paths return well-formed predictions; none can produce a 5xx
or a skip.

## Known issues

- **Cache writer race in the LLM tier.** When the same `(curr, prior)` pair
  appears in multiple cases that run concurrently, both LLM calls fire before
  either writes the cache, and the last writer wins. Combined with mild
  non-determinism in OpenAI at `temperature=0.0`, repeat warm-cache hybrid
  runs can disagree on ~24 / 27,614 (≈ 0.09 %) pairs. Not relevant in normal
  operation now that the classifier is the primary tier; left in place because
  the LLM is fallback-only.
- **`OPENAI_API_KEY` failure modes are silent at the request level.** During
  development we hit a 401 from a revoked key and the system silently fell
  through to heuristic. Worth adding a startup-time `models.list()` smoke
  check that surfaces the failure loudly. Less urgent now that the
  classifier doesn't need the LLM.
- **LightGBM has small non-determinism** from `bagging_fraction` /
  `feature_fraction` sampling without a fixed seed. Two training runs of the
  same script can land in 0.9595–0.9605 range. Add `seed=42` to the params
  for fully reproducible experiments.

## Next steps and methodologies worth trying

Ranked by my honest read of expected gain × effort. The deployed classifier
is at 0.9603 OOF; 0.97+ probably requires multiple of these to compose.

### Quick wins (a few hours of work each)

1. **Hand-curate a "tricky pairs" lookup table.** The remaining ~1,100 errors
   come from <50 description-pair clusters (e.g., echo↔chest CT, CT-angio
   variants, bone scan↔specific organ). One round of hand-labeling those and
   bypassing both classifier and LLM for them would tighten accuracy at zero
   runtime cost. Expected: +0.3 to +0.7 pp.

2. **Probability calibration + threshold sweep.** The classifier's threshold
   is fixed at 0.5 but the OOF sweep showed 0.40 nudges accuracy slightly
   higher (0.9606). Worth a proper validation-set sweep once private-split
   labels are available. Expected: +0.05 pp.

3. **Bigram / trigram token features.** Add character n-gram cosine similarity
   between descriptions to the feature matrix. Captures sub-word similarity
   the regex parser misses (e.g., "LIMITED" vs "LMTD"). Expected: +0.1 pp.

4. **Modality-conditional cross-region rules.** The 13 dropped blanket links
   were structurally wrong, but some likely survive when conditioned on
   modality. Test "heart ↔ chest under CT/MRI", "carotid ↔ brain under MRA",
   etc., and add them as features. Expected: +0.1 to +0.3 pp.

5. **Patient-level features from `patient_name` / `patient_id`.** Currently
   ignored. Could surface signal like "this patient has a long-term cancer
   workup, so PET priors are more relevant." Expected: depends entirely on
   whether the IDs encode anything useful — start with a sniff test.

### Larger investments (a day or more)

1. **Fine-tune a small encoder (DistilBERT / MiniLM) on labeled pairs.**
   Treat each `(curr, prior)` pair as a sentence-pair classification task,
   fine-tune for 2–3 epochs. Cross-encoder architecture with a binary head
   on top of `[CLS]`. Public-split has 27 k labeled pairs — enough for a
   small model. Expected: +0.3 to +0.7 pp if executed well, but adds a
   significant model + tokenizer dependency to the Docker image. Could be
   an alternative tier 1 (with the LightGBM as fallback) or used as a
   stacked feature into LightGBM.

2. **Domain-specific embeddings.** Replace MiniLM with BioBERT,
   ClinicalBERT, or BioLinkBERT and re-test the cosine + summary-features
   experiment. These models were trained on PubMed / clinical notes and
   should know that "MAM US BI breast screening" and "MG tomo screening
   bilateral" are clinical synonyms. If even one of these moves the OOF
   number up by 0.1+ pp, the embedding feature is worth the dependency.
   Expected: +0.2 to +0.5 pp; size cost ~1 GB.

3. **Self-consistency / ensembling on borderline pairs.** For pairs where
   the classifier reports `proba ∈ [0.4, 0.6]` (genuinely uncertain), fan
   out to N=3 calls of `gpt-4o-mini` with different temperatures, take
   majority vote. Combine with the classifier prediction via stacked
   logistic regression. Adds latency for ~5 % of pairs but only on those
   the classifier is unsure about. Expected: +0.2 to +0.4 pp.

4. **Active-learning loop.** Use the classifier's OOF probability to surface
   the highest-uncertainty pairs from the public split, hand-label them
   precisely, retrain. Iterate 2–3 rounds. Lifts accuracy on the same
   description-pair clusters that are currently borderline. Expected:
   +0.3 to +0.5 pp; significant manual effort.

5. **Stacked ensemble.** Train a second classifier (e.g., XGBoost with
   different hyperparams, or a logistic regression on a different feature
   cut) and combine via meta-learner. Modest gain on tabular tasks
   historically. Expected: +0.05 to +0.15 pp.

### Architectural changes worth considering

1. **Replace the LLM tier entirely** by removing `app/llm.py` and the OpenAI
   dependency from the Docker image. The classifier doesn't need it. This is
   a code-cleanup win, not an accuracy win — but reduces failure surface,
   image size, and submission-package complexity. Worth doing once the
   classifier path is fully validated against the private split.

2. **Per-modality model partitioning.** Train separate LightGBM models for
   `(CT, CT)`, `(MRI, MRI)`, `(XR, XR)`, etc. Each model focuses on its
   modality's failure patterns. Routing logic at inference. Probably
   overkill for 27 k examples but worth thinking about if data grows.

3. **Calibrated cost-sensitive training.** Right now we treat false-positives
   and false-negatives equally because the metric is symmetric accuracy. If
   the scoring ever changes to reward precision or recall asymmetrically,
   LightGBM's `is_unbalance` / `scale_pos_weight` parameters give us a
   direct knob.

### What I'd specifically *not* recommend

- **Larger LLM (`gpt-4o`, `gpt-5`).** Hybrid v1 → hybrid v2 (prompt change)
  was already within noise. The bottleneck isn't the model's reasoning, it's
  the inherent label ambiguity in short text.
- **General-purpose sentence embeddings on top of engineered features.**
  Already tested in Option 3a/3b — flat result.
- **More heuristic rules from domain intuition.** 13/13 ablations
  net-negative; the dataset structurally rejects this approach.
- **Reasoning-mode LLMs (`o1-mini`, `o3-mini`).** Per-pair latency would blow
  the 360 s budget on the worst-case 234-prior case.
