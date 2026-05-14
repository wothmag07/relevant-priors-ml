# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HTTP API for the **relevant-priors-v1** challenge. Given a current radiology
exam and a list of previous exams for the same patient, predict for each prior
whether it should be shown to the radiologist. Final scoring uses a private
split; we only have the public split (`relevant_priors_public.json`, 996
cases / 27,614 labeled pairs) for local iteration.

`README.md` describes the public-facing API. `EXPERIMENT.md` is the experiments +
next-steps document required as a submission artifact — keep it in sync with
real measured numbers.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env                    # then set OPENAI_API_KEY for the LLM tier

pytest tests/ -q                        # contract tests (no API key needed)
pytest tests/test_contract.py::test_predict_contract_shape -q   # single test

python -m eval.run_eval --predictor always_false           # 76.22% sanity floor
python -m eval.run_eval --predictor heuristic              # ~94% (no LLM)
LLM_ENABLED=1 python -m eval.run_eval --predictor hybrid   # heuristic + gpt-4o-mini
python -m eval.run_eval --predictor heuristic --limit 50 --errors 30   # debug subset

uvicorn app.main:app --reload           # dev server on :8000
```

There is no linter or formatter wired up. If you add one, document it here.

## Architecture (read this before changing classifier behavior)

Two-tier hybrid. The contract handler (`app/main.py`) calls `predict_cases_async`
which runs both tiers:

1. **Heuristic** (`app/parser.py` → `app/heuristic.py`)
   `parse_description()` runs ordered regex rules over a normalized
   description and returns a `StudyTags(regions, modality, contrast, laterality,
   is_outside)`. `classify_pair()` returns
   `HeuristicResult(predicted, confidence, reason)`. Relevance is region-set
   overlap, with `COVERAGE_EXPANSIONS` (e.g. `abd_pel → {abdomen, pelvis}`,
   `wholebody → torso regions`) and a tiny `RELATED_REGION_PAIRS` set.
   `EXCLUSIVE_REGION_TAGS` (currently `bone_density`, `eeg`) override all
   other regions when present — this is how DXA stops matching `hip` or
   `spine`. Order in `REGION_PATTERNS` matters: more specific patterns must
   come first (e.g. `HEAD AND NECK → neck` before bare `HEAD → brain`).

2. **LLM** (`app/llm.py` → `app/classifier.py`)
   Pairs with heuristic confidence `< AMBIGUOUS_THRESHOLD` (0.85) escalate to
   `gpt-4o-mini`. Batching is **one call per case**, not per pair and not per
   request — see "Batching" below. Calls run concurrently with an asyncio
   semaphore (`LLM_MAX_CONCURRENCY`, default 8). Results are cached on disk in
   `.cache/llm_cache.json` keyed by `(curr_description, prior_description)`.
   Dates and patient identity are intentionally NOT in the cache key.

Same code path is used by both the FastAPI handler and the eval harness, so
local accuracy = production accuracy.

## Hard rules (don't break these)

- **The response contract is exact.** Every prior in the request gets one
  prediction in the response with the same `case_id` and `study_id`. Skipped
  predictions count as incorrect — do not filter, sort, or deduplicate. The
  contract test in `tests/test_contract.py` enforces shape.
- **Never let a request 5xx.** `app/main.py::_handle` catches classifier
  exceptions and returns one all-False prediction per prior (the 76% baseline)
  rather than letting the request fail. If you add new code in the request
  path, preserve this. Skipping is worse than guessing.
- **Cache key is `(curr_desc, prior_desc)` only.** Don't add dates or
  patient identity to the cache key — it would explode cache size and the
  signal lives in the descriptions.
- **Tune by data, not by domain intuition.** Several "obviously right"
  linkages (heart↔chest, brain↔sinuses, t_spine↔chest, OUTSIDE FILMS →
  relevant) are net-negative on the public split and were intentionally
  removed. Before adding or removing a rule in `parser.py` or
  `heuristic.py`, count `(positive, negative)` pairs in the public truth
  and only add it when it's clearly net-positive. The
  ablations table in `EXPERIMENT.md` records what's been tested.
- **`OUTSIDE FILMS` defaults to `False`.** Counterintuitive but data is
  unanimous (0 / 67 in the public split). Don't "fix" this without checking.

## Batching

The brief explicitly says one LLM call per *examination* will time out
(360 s evaluator budget). The implementation batches per *case*: one call
covers one current study + all of its ambiguous priors. Don't reduce
granularity to per-pair. Don't increase to per-request without a token-budget
guard — single cases can have up to 234 priors, and a request can carry
hundreds of cases.

## Deployment

Hugging Face Space, Docker SDK, port 7860 (set in `README.md` frontmatter
and `Dockerfile`). The Dockerfile also `COPY`s `.cache/` if present so a
warmed cache ships with the image and the deployed endpoint hits zero LLM
calls for previously-seen description pairs. `OPENAI_API_KEY` must be set as
a Space secret.

The challenge submission requires three things: endpoint URL, code zip, and
`EXPERIMENT.md`. When measurements change, update both `EXPERIMENT.md` (experiments
table) and any inline numbers in `README.md`.

## Working with this codebase

When iterating on the heuristic, the productive loop is:

1. Run `python -m eval.run_eval --predictor heuristic --errors 30`.
2. Pick a top error cluster.
3. Count its `(positive, negative)` distribution in the truth (one-off
   `python -c` is fine; there are examples in earlier commit messages).
4. Add/remove a rule only if data supports it.
5. Re-run eval; record the delta in `EXPERIMENT.md` if the change ships.

When iterating on the LLM tier, prefer adjusting the system prompt in
`app/llm.py` or the threshold in `app/classifier.py` over adding more
heuristic rules — the LLM exists precisely to handle the long tail.
