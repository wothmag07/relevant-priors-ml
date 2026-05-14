"""Pair-relevance classifier with two layered tiers.

Tier order:
1. **LightGBM classifier** (`app/classifier_model.py`) — primary path. Trained
   offline on the public split with engineered features (region overlap,
   modality, contrast, laterality, date deltas, description text features,
   plus the heuristic's own prediction as a feature). Deterministic, fast,
   no external API dependency.
2. **Heuristic only** (`app/parser.py`, `app/heuristic.py`) — fallback when
   the LightGBM model isn't available (e.g., model file missing on a deploy
   or feature-schema drift detected at load time).
3. All-False fallback — handled at the FastAPI layer in `app/main.py`.

Public entry points:
    predict_cases(cases) -> list[Prediction]              # sync wrapper
    predict_cases_async(cases) -> list[Prediction]
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from app.classifier_model import is_available as _classifier_available
from app.classifier_model import predict_batch as _classifier_predict
from app.heuristic import classify_pair
from app.parser import parse_description
from app.schemas import Case, Prediction

logger = logging.getLogger(__name__)


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
    assert preds is not None, "_predict_via_classifier called when classifier unavailable"  # noqa: S101
    return [
        Prediction(case_id=k[0], study_id=k[1], predicted_is_relevant=p)
        for k, p in zip(keys, preds, strict=True)
    ]


def _predict_via_heuristic(cases: list[Case]) -> list[Prediction]:
    """Tier 2: heuristic-only fallback. Used when the classifier model is
    missing or its feature schema doesn't match the current code."""
    predictions: list[Prediction] = []
    for c in cases:
        curr_tags = parse_description(c.current_study.study_description)
        for p in c.prior_studies:
            prior_tags = parse_description(p.study_description)
            r = classify_pair(curr_tags, prior_tags)
            predictions.append(
                Prediction(
                    case_id=c.case_id,
                    study_id=p.study_id,
                    predicted_is_relevant=r.predicted,
                )
            )
    return predictions


async def predict_cases_async(cases: Iterable[Case], request_id: str = "-") -> list[Prediction]:
    cases = list(cases)
    total_priors = sum(len(c.prior_studies) for c in cases)

    if _classifier_available():
        predictions = _predict_via_classifier(cases)
        tier = "classifier"
    else:
        predictions = _predict_via_heuristic(cases)
        tier = "heuristic"

    logger.info(
        "request_id=%s predict cases=%d priors=%d tier=%s",
        request_id, len(cases), total_priors, tier,
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
