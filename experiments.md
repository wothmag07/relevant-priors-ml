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

Two layered tiers in `app/classifier.py`. The first to succeed wins.

1. **LightGBM classifier** (`app/classifier_model.py`, `app/features.py`) —
   primary path. Trained offline on the public split with ~120 engineered
   features per pair: region one-hots, modality, contrast, laterality, date
   deltas, description-text features, and the heuristic's own prediction +
   confidence as inputs. Deterministic, ~2 ms/pair, no API dependency at
   inference. Trained with `seed=42` + `deterministic=True` so two runs
   produce identical fold accuracies.
2. **Heuristic-only fallback** (`app/parser.py`, `app/heuristic.py`) — fires
   if `classifier_model.pkl` is missing on a deploy or if a
   feature-schema-drift check at load time refuses the model. Pure regex +
   region-set overlap, no learned component.
3. **All-False fallback** (`app/main.py`) — final safety net at the FastAPI
   layer. If both tiers above fail or the request exceeds a 300 s wall-clock
   budget, one all-False prediction per prior is returned (76 % baseline).
   No prediction is ever skipped.

The wall-clock budget is enforced with `asyncio.wait_for(..., timeout=300)`;
on expiry, in-flight tasks are cancelled and tier 3 fires. This is the hard
guarantee against the evaluator's 360 s timeout regardless of what happens
upstream.

## Pipeline

Two phases that share the same feature-engineering code in
`app/features.py`, which guarantees train/inference parity.

### Training (offline, one-shot)

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
GroupKFold (groups = unique curr+prior text pair, stable across runs)
   → 5-fold OOF CV (deterministic, seed=42)
        │
        ▼
lgb.train(full_data) → app/classifier_model.pkl    (1 MB pickle)
```

### Inference (online, every `POST /predict` request)

```text
POST /predict {cases: [...]}
        │
        ▼
app/main.py::_handle      ← wraps in 300 s wall-clock budget
        │
        ▼
app/classifier.py::predict_cases_async
        │
        ▼
┌──────────────────────────────────┐
│ Tier 1: LightGBM                 │ ← fires if classifier_model.pkl is present
│   featurize each pair            │   AND its feature schema matches features.py
│   model.predict(X)               │
│   bool[i] = proba[i] >= 0.5      │
└──────────┬───────────────────────┘
           │ model missing or schema drift? ↓
┌──────────▼───────────────────────┐
│ Tier 2: heuristic-only           │ ← parser → classify_pair → bool
└──────────┬───────────────────────┘
           │ tier 2 raises or budget expires? ↓
┌──────────▼───────────────────────┐
│ Tier 3: all-False                │ ← never skips, returns 76 % baseline
└──────────────────────────────────┘
        │
        ▼
HTTP 200 {predictions: [...]}
```

## Experiments

All numbers below are on the full public split (996 cases / 27,614 pairs).
The classifier row reports **out-of-fold accuracy from 5-fold GroupKFold CV**
(groups = unique description-pair strings) — the unbiased generalisation
estimate. Training-set fit is also shown for transparency but is not what
we claim.

| # | Predictor | Accuracy | Δ vs always-False | Wall time |
| --- | --- | --- | --- | --- |
| 0 | `always_false` (sanity floor) | 0.7622 | — | 0.03 s |
| 1 | Heuristic v1 (initial parser, all RELATED pairs on) | 0.9208 | +15.86 pp | 3.1 s |
| 2 | Heuristic v2 (data-validated related pairs only) | 0.9383 | +17.61 pp | 3.1 s |
| 3 | Heuristic v3 (`ABD_PEL` regex fix, `OUTSIDE FILMS=False`, ultrasound-breast-screening tag) | 0.9408 | +17.86 pp | 3.0 s |
| 4 | Heuristic v3.1 (`LEGS` plural, `NMmyo` concat, span-priority suppression, `HEAD AND NECK` literal) | 0.9418 | +17.96 pp | 3.0 s |
| 5 | Hybrid (heuristic v3 + `gpt-4o-mini` for conf < 0.85), cold cache | 0.9491 | +18.69 pp | 186 s |
| 5w | Hybrid, warm cache (zero LLM calls) | 0.9482 | +18.60 pp | 1.7 s |
| 6 | Hybrid + revised LLM prompt (explicit conservative defaults + edge-case exclusions) | 0.9493 | +18.71 pp | 175 s |
| **7** | **LightGBM classifier (engineered features), 5-fold OOF** | **0.9603** | **+19.81 pp** | **65 s** |
| 7t | LightGBM, training-set fit on full public split | 0.9798 | — | 65 s |
| 8a | LightGBM + sentence-embedding cosine similarity | 0.9594 | +19.72 pp | 70 s |
| 8b | LightGBM + cosine + diff/sum/prod summary stats | 0.9594 | +19.72 pp | 75 s |

Row 7 (the shipped classifier) is reproducible across runs. We discovered
during development that `hash((curr_desc, prior_desc))` is a randomised
group identifier (Python's `hash()` uses `PYTHONHASHSEED` by default), so
the GroupKFold split was unstable across processes. Switching to the
joined description-pair string and pinning `seed=42` /
`deterministic=True` / `n_jobs=1` in the LightGBM params produces
byte-identical fold accuracies (`0.9622, 0.9553, 0.9643, 0.9649, 0.9547`)
across runs.

## Confusion matrices

### Heuristic v3.1 (deployed fallback)

```text
              pred False   pred True
true False        20497         550
true True          1056        5511
```

Failure modes are roughly symmetric (550 false-positives, 1,056
false-negatives). Remaining errors are dominated by clusters that are
intrinsically ambiguous from text alone: echo ↔ chest CT/XR, CT angio
head ↔ CT angio carotid, EEG ↔ MRI brain, CT-guided FNA, bone scan ↔
specific organ studies.

### LightGBM classifier (5-fold OOF, seeded)

```text
              pred False   pred True
true False        20659         388
true True           709        5858
```

Compared with heuristic v3.1: **+347 TP**, **−162 FP**, **−347 FN**,
**+162 TN** — both error axes improve. The +509 net correctly
classified pairs translate to the +1.85 pp improvement over
heuristic-only.

## What we tried — heuristic side

### Parser-level changes that helped

| Change | Public-split evidence | Decision |
| --- | --- | --- |
| `bone_scan → wholebody` (covers torso PET/NM scans) | dominates bone-scan TPs | added |
| `ULTRASOUND BILAT SCREEN → breast` | 107 / 4 | added |
| `OUTSIDE FILMS → False` | 0 / 67 | added |
| `bone_density` exclusive (suppresses `hip`/`spine` co-tags) | 41 / 575 cross-pairs | added |
| Underscore in `ABD_PEL` regex | bug — `\W` excludes `_` in Python | fixed |
| `\b(LEG\|LEGS\|LE)\b` (catch `BILAT LEGS`) | 213 occurrences, +5 heuristic predictions | fixed |
| `\bN?M\s*MYO\s*PERF\b` (catch `NMmyo perf`) | 70 occurrences, +3 heuristic predictions | fixed |
| Span-priority suppression in `parse_description` | bug — `HEAD AND NECK` was tagging as both `neck` AND `brain` | fixed |
| `\bHEAD\s+AND\s+NECK\b` literal alternation | bug — `\W+` doesn't match the letters in " AND " | fixed |

### Cross-region pair ablations — every candidate net-negative

We tested adding cross-region links to `RELATED_REGION_PAIRS`. The pattern
is consistent: labelers in this dataset are judging *clinical utility*
(does this prior add value to today's read?), not anatomical overlap.
Anatomically-adjacent pairs are mostly labeled negative.

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

Thirteen candidates, all net-negative. Strong evidence that blanket
cross-region rules don't fit this dataset; any future linkages should be
**modality-conditional**, not region-only.

## What we tried — LLM side

### Hybrid heuristic + `gpt-4o-mini`

For pairs with heuristic confidence `< 0.85`, we batched all ambiguous
priors per case into a single OpenAI call (one HTTP round-trip per case,
not per pair) with `response_format=json_object` and cached results on
disk by `(curr_desc, prior_desc)`. Cold-cache pass over the public split:
1,978 unique ambiguous pairs, ~$0.10–0.20 in API cost, 186 s wall time.
Warm cache made every repeat run 0 s of LLM cost.

Result: **0.9491 cold / 0.9482 warm** — +0.83 pp over heuristic v3, but
beneath the LightGBM classifier and at the cost of an external API
dependency.

### Revised LLM prompt

We rewrote the system prompt to add explicit conservative-default rules
("when uncertain, predict NOT relevant — labelers are strict") plus
data-validated edge-case exclusions (spine ↔ chest/abdomen/pelvis,
carotid ↔ brain, EEG ↔ brain, GI fluoro ↔ chest). Result: 0.9491 →
**0.9493**, within the cache-race noise floor (~24 predictions). The
directional shift in the confusion matrix was the right one (more
conservative — fewer FPs, more TNs) but the magnitude was negligible.

### Decision: drop the LLM tier

Once the LightGBM classifier was in place, the LLM tier became dead code
in normal operation — it only fired when the model file was missing. We
deleted `app/llm.py`, `.cache/`, the `openai` dependency, and the LLM
tier-routing in `classifier.py`. The new fallback is heuristic-only
(0.9418), which fires only on a deploy where the model file is absent
or has drifted from the current `features.py`.

Result: smaller image (~10 MB lighter), no external API dependency at
inference, no silent-failure mode from a misconfigured key.

## What we tried — classifier side

### Engineered-feature LightGBM (Option 2)

121 features per pair across:

- **Region one-hots** for the 20 most-common region tags (curr / prior /
  both)
- **Direct + expanded region overlap** indicators (using
  `COVERAGE_EXPANSIONS`)
- **Modality** one-hots (CT, MRI, XR, etc.) per side, plus
  `same_modality` and `both_cross_section`
- **Contrast match** (with / without / mixed) and **laterality
  mismatch** (left vs right)
- **Date delta** in days, log-scaled, and bucketed (`<30d`, `<1y`,
  `<2y`, `>5y`)
- **Description-text features**: lengths, length ratio, shared-token
  count
- **Heuristic prediction + confidence + reason** (the heuristic's output
  is itself a feature, so the model can learn when to trust or override
  it)

Top features by gain:

| Rank | Feature | Notes |
| --- | --- | --- |
| 1 | `heur_pred` | Dominant input — the model is essentially "augmented heuristic" |
| 2 | `expanded_overlap` | Region overlap with coverage expansions |
| 3 | `date_delta_days` | Signal the heuristic ignored — adds real lift |
| 4 | `heur_conf` | Heuristic confidence as a second-order signal |
| 5 | `lateral_mismatch` | Parsed but never used by the heuristic |
| 6 | `direct_overlap` | Literal region-set intersection |
| 7 | `shared_tokens` | Token-level text similarity |
| 8 | `desc_len_ratio` | Description length ratio |

The classifier essentially **augments** the heuristic with date deltas,
laterality, and text-level similarity — three signals the heuristic
doesn't look at. That's where the +1.85 pp jump comes from.

### Sentence embeddings (Option 3) — no improvement

We tested whether MiniLM-based embeddings could lift the classifier
beyond engineered features.

| Variant | Features | OOF acc | Δ vs Option 2 |
| --- | --- | --- | --- |
| Option 2 (engineered only) | 121 | 0.9595 | — |
| Option 3a (+ embedding cosine sim) | 122 | 0.9594 | −0.01 pp |
| Option 3b (+ cosine + diff/sum/prod summaries) | 130 | 0.9594 | −0.01 pp |

All within noise. Three reasons embeddings don't help here:

- Descriptions are short bureaucratic strings, not natural language.
- Engineered features already capture the same content (regions,
  modality, shared tokens).
- MiniLM is general-purpose and lacks radiology-domain knowledge — the
  100-line regex parser is at a knowledge advantage on these specific
  terms.

The `sentence-transformers` / `torch` dependencies were uninstalled
after this measurement. A *medical-domain* embedding (BioBERT,
ClinicalBERT, RadBERT) might do better, but the dependency cost
(~1 GB image growth) wasn't justified by the data.

## Test coverage

The codebase has two test files:

- `tests/test_contract.py` (3 tests) — feeds the example payload from
  the challenge brief through the FastAPI endpoint and verifies request
  / response shapes match the spec, including all paths (`/predict`,
  `/`, `/healthz`).
- `tests/test_parser.py` (27 tests) — pins parser and heuristic edge
  cases:
  - `ABD_PEL` underscore / slash / space forms
  - `OUTSIDE FILMS` flag handling and the "OUTSIDE SCREENING US BREAST
    BILATERAL" exception (which is *not* an outside-films pattern)
  - `HEAD AND NECK` priority over bare `HEAD`
  - Exclusive tags: `bone_density` and `eeg` overriding co-tags
  - Laterality (left / right / bilateral), modality detection (CT, MRI
    not MRA, CTA precedence), contrast (`with_without`)
  - Cross-region exclusions: `vasc_carotid ↮ brain`, `eeg ↮ brain MRI`,
    `heart ↮ chest`, `t_spine ↮ chest`, `l_spine ↮ abdomen`
  - Coverage overlaps that *do* fire: `abd_pel ↔ abdomen`,
    `wholebody ↔ chest`
  - `classify_pair` confidence + reason wiring per branch

Total: 30 tests, run in ~0.3 s. All decisions called out in this
document have a corresponding test that would fail if the rule were
silently regressed.

## Cost & latency

| Predictor | Wall time on full public split | API cost | External deps |
| --- | --- | --- | --- |
| Heuristic only | 3 s | $0 | none |
| Hybrid (cold cache, deprecated) | 186 s | ~$0.10–0.20 | OpenAI |
| Hybrid (warm cache, deprecated) | 1.7 s | $0 | OpenAI |
| **LightGBM classifier (deployed)** | **65 s** | **$0** | **none — model packaged in image** |

Per-pair: ~2.4 ms for the classifier. Worst-case private-split case
(234 priors) → ~0.6 s for that one case. A request with 100 such cases
→ ~1 min, comfortably inside the 360 s evaluator budget.

If `classifier_model.pkl` is missing on a deploy, or the saved model's
feature schema doesn't match `app/features.py` (drift check at load
time), the system silently degrades to **heuristic-only** (0.9418).
The all-False fallback at the FastAPI layer is the ultimate floor
(0.7622). All three return well-formed predictions; none can produce a
5xx or a skip.

## Reproducibility

The training pipeline is fully deterministic:

- `eval/train_classifier.py` uses the joined description-pair string
  (not Python's `hash()`) as the GroupKFold group identifier, so folds
  are stable across processes.
- LightGBM is trained with `seed=42`, `feature_fraction_seed=42`,
  `bagging_seed=42`, `data_random_seed=42`, `deterministic=True`, and
  `n_jobs=1`. Two consecutive runs produce byte-identical fold
  accuracies and OOF totals.
- `app/classifier_model.py` validates the saved model's
  `feature_names` against the live `feature_names()` from
  `app/features.py` at load time. On mismatch, the classifier tier
  disables itself loudly and the system falls through to heuristic —
  preventing silent garbage from a feature-schema drift.
- Reproduction: `python -m eval.train_classifier` prints fold and OOF
  numbers; `python -m eval.train_classifier --save` rebuilds the
  pickle.

## Next steps

Ranked by my read of expected gain × effort. The deployed classifier is
at 0.9603 OOF on the public split.

### Quick wins (a few hours each)

1. **Hand-curate a "tricky pairs" lookup table.** The remaining ~1,100
   errors come from <50 description-pair clusters. One round of
   hand-labeling those and bypassing the classifier for them would
   tighten accuracy at zero runtime cost. Expected: +0.3 to +0.7 pp.
2. **Probability calibration + threshold sweep.** The classifier's
   threshold is fixed at 0.5 but the OOF sweep showed 0.40 nudges
   accuracy slightly higher (0.9606). Worth a proper sweep with
   stratified validation. Expected: +0.05 pp.
3. **Bigram / trigram token features.** Add character n-gram cosine
   similarity between descriptions to the feature matrix. Captures
   sub-word similarity the regex parser misses (e.g. "LIMITED" vs
   "LMTD"). Expected: +0.1 pp.
4. **Modality-conditional cross-region rules.** The 13 dropped blanket
   links were structurally wrong, but some likely survive when
   conditioned on modality. Test "heart ↔ chest under CT/MRI",
   "carotid ↔ brain under MRA", etc., and add them as features.
   Expected: +0.1 to +0.3 pp.
5. **Patient-level features.** `patient_name` / `patient_id` are
   currently ignored. Could surface signal like "this patient has a
   long-term cancer workup, so PET priors are more relevant." Start
   with a sniff test on the public split.

### Larger investments (a day or more)

1. **Fine-tune a small encoder** (DistilBERT / MiniLM) on labeled pairs
   as a sentence-pair binary classifier. 27 k labeled pairs is enough
   for a small cross-encoder; could be tier 1 or a stacked feature into
   LightGBM. Expected: +0.3 to +0.7 pp; cost: ~200 MB image growth.
2. **Domain-specific embeddings** (BioBERT, ClinicalBERT, BioLinkBERT).
   These were trained on PubMed / clinical notes and should know
   radiology synonyms our regex parser doesn't. Re-test the
   cosine-summary experiment with one of them. Expected: +0.2 to
   +0.5 pp; cost: ~1 GB image growth.
3. **Self-consistency on borderline pairs.** For pairs where the
   classifier's `proba ∈ [0.4, 0.6]`, fan out to N=3 calls with
   different prompts / models, take majority vote. Expected: +0.2 to
   +0.4 pp.
4. **Active-learning loop.** Use the classifier's OOF probability to
   surface the highest-uncertainty pairs from the public split,
   hand-label them precisely, retrain. Iterate 2–3 rounds. Expected:
   +0.3 to +0.5 pp.
5. **Stacked ensemble.** Train a second classifier with different
   hyperparameters or feature cuts, combine via meta-learner.
   Expected: +0.05 to +0.15 pp.

### Radiologist-centred validation

The data-driven nature of all tuning above means our exclusions and
inclusions reflect *what the labelers in this dataset chose*, not
necessarily *what a working radiologist would prefer for their actual
workflow*. To validate the model in a clinical setting:

1. **Top-cluster review.** Pull the 30 highest-volume description-pair
   clusters where the classifier and the labels disagree (in either
   direction). Examples expected: echo↔chest CT, CT-angio
   head↔CT-angio carotid, EEG↔brain MRI, l_spine↔abdomen,
   bone-scan↔organ-specific.
2. **Blind clinician audit.** Present each cluster's description pair
   to 2–3 staff radiologists as: "If you were reading [current], would
   you want to see [prior]? Yes / No / It depends — explain." No
   labels shown.
3. **Disagreement triage.** For pairs where clinicians split or
   disagree with both the model and the labels, flag as
   intrinsically ambiguous; consider returning a calibrated probability
   rather than a hard bool.
4. **Workflow-aware features.** Talk to clinicians about features the
   parser ignores: sub-specialty (cardiac vs general), shift context,
   patient age cohort. Some are inferable from the description; others
   would require expanding the request schema.
5. **Calibration over time.** In a real RIS deployment, log the model's
   probability, the radiologist's chosen action, and whether the prior
   was actually opened. This lets us measure "did the prior add value
   when shown" rather than only "was the bool correct."

### Things I'd specifically *not* try

- **Larger general-purpose LLMs** (`gpt-4o`, `gpt-5`). The hybrid prompt
  rewrite was within noise; the bottleneck isn't model reasoning, it's
  inherent label ambiguity in short text.
- **General-purpose sentence embeddings** on top of engineered features
  — already tested in Option 3a/3b, flat result.
- **More heuristic rules from domain intuition.** 13/13 cross-region
  ablations were net-negative; the dataset structurally rejects this
  approach.
- **Reasoning-mode LLMs** (`o1-mini`, `o3-mini`). Per-pair latency would
  blow the 360 s budget on the worst-case 234-prior case.
