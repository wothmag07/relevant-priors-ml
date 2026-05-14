"""Pair-relevance classifier with three layered tiers.

Tier order:
1. **LightGBM classifier** (`app/classifier_model.py`) — primary path. Trained
   offline on the public split with engineered features (region overlap,
   modality, contrast, laterality, date deltas, description text features,
   plus the heuristic's own prediction as a feature). Deterministic and fast.
2. **Heuristic + LLM** — fallback when the LightGBM model isn't available
   (e.g., model file missing on a deploy). Same code path as before.
3. All-False fallback — handled at the FastAPI layer in `app/main.py`.

Public entry points:
    predict_cases(cases) -> list[Prediction]              # sync wrapper
    predict_cases_async(cases) -> list[Prediction]
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable

from app.classifier_model import is_available as _classifier_available, predict_batch as _classifier_predict
from app.heuristic import HeuristicResult, classify_pair
from app.llm import classify_batch_async, get_cache, llm_enabled
from app.parser import parse_description
from app.schemas import Case, Prediction

logger = logging.getLogger(__name__)

# Heuristic results with confidence < this threshold are escalated to the LLM.
# (Only used when the LightGBM classifier is unavailable.)
AMBIGUOUS_THRESHOLD = 0.85


def _predict_via_classifier(cases: list[Case]) -> list[Prediction]:
    """Tier 1: LightGBM classifier on every (curr, prior) pair."""
    pairs = []
    keys = []
    for c in cases:
        ct = parse_description(c.current_study.study_description)
        for p in c.prior_studies:
            pt = parse_description(p.study_description)
            pairs.append((
                c.current_study.study_description,
                p.study_description,
                c.current_study.study_date,
                p.study_date,
                ct,
                pt,
            ))
            keys.append((c.case_id, p.study_id))
    preds = _classifier_predict(pairs)
    return [
        Prediction(case_id=k[0], study_id=k[1], predicted_is_relevant=p)
        for k, p in zip(keys, preds)
    ]


async def predict_cases_async(cases: Iterable[Case], request_id: str = "-") -> list[Prediction]:
    cases = list(cases)
    total_priors = sum(len(c.prior_studies) for c in cases)

    # Tier 1: LightGBM classifier (preferred when available).
    if _classifier_available():
        predictions = _predict_via_classifier(cases)
        logger.info(
            "request_id=%s predict cases=%d priors=%d tier=classifier",
            request_id, len(cases), total_priors,
        )
        return predictions

    # Tier 2: heuristic + LLM fallback (used only if model file is missing).
    pre: list[tuple[Case, list[HeuristicResult], list[int], list[str]]] = []
    total_ambig = 0
    for case in cases:
        curr_tags = parse_description(case.current_study.study_description)
        prior_tags = [parse_description(p.study_description) for p in case.prior_studies]
        results = [classify_pair(curr_tags, pt) for pt in prior_tags]
        ambig_idxs = [i for i, r in enumerate(results) if r.confidence < AMBIGUOUS_THRESHOLD]
        ambig_priors = [case.prior_studies[i].study_description for i in ambig_idxs]
        pre.append((case, results, ambig_idxs, ambig_priors))
        total_ambig += len(ambig_idxs)

    use_llm = llm_enabled() and total_ambig > 0
    if use_llm:
        max_conc = int(os.environ.get("LLM_MAX_CONCURRENCY", "8"))
        sem = asyncio.Semaphore(max_conc)

        async def run_case(idx: int):
            case, _results, ambig_idxs, ambig_priors = pre[idx]
            if not ambig_priors:
                return idx, []
            llm_out = await classify_batch_async(
                case.current_study.study_description,
                ambig_priors,
                semaphore=sem,
            )
            return idx, llm_out

        gathered = await asyncio.gather(*(run_case(i) for i in range(len(pre))))
        try:
            get_cache().flush()
        except Exception as e:
            logger.warning("cache flush failed: %s", e)
    else:
        gathered = [(i, []) for i in range(len(pre))]

    # Stitch predictions.
    predictions: list[Prediction] = []
    llm_overrides: dict[int, list] = dict(gathered)

    for idx, (case, results, ambig_idxs, _ambig_priors) in enumerate(pre):
        llm_results = llm_overrides.get(idx) or []
        # Map ambiguous index -> LLM result
        ambig_by_pos = {ai: lr for ai, lr in zip(ambig_idxs, llm_results)}
        for i, prior in enumerate(case.prior_studies):
            heur = results[i]
            llm = ambig_by_pos.get(i)
            if llm is not None and llm.source != "fallback":
                predicted = llm.predicted
            else:
                predicted = heur.predicted
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=predicted,
                )
            )

    logger.info(
        "request_id=%s predict cases=%d priors=%d ambiguous=%d llm_enabled=%s tier=heuristic+llm",
        request_id, len(cases), total_priors, total_ambig, use_llm,
    )
    return predictions


def predict_cases(cases: Iterable[Case]) -> list[Prediction]:
    """Sync wrapper for use from non-async contexts (e.g., the eval harness)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an event loop already (FastAPI handler) — caller
            # should have used the async entry point. Fall back gracefully.
            raise RuntimeError("predict_cases called from within a running loop; use predict_cases_async")
    except RuntimeError:
        # No running loop, OK to use asyncio.run
        pass
    return asyncio.run(predict_cases_async(cases))
