"""Inference-time wrapper around the trained LightGBM (curr, prior) classifier.

Lazy-loads the pickled model from disk on first use. Returns None if the model
file is missing — callers should fall back to the heuristic / LLM path in
that case so the service never crashes when the model isn't shipped.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from pathlib import Path

import numpy as np

from app.features import featurize
from app.parser import StudyTags

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent / "classifier_model.pkl"

_lock = threading.Lock()
_loaded = False
_model = None
_feature_names: list[str] = []
_threshold: float = 0.5


def _load_once() -> bool:
    global _loaded, _model, _feature_names, _threshold
    if _loaded:
        return _model is not None
    with _lock:
        if _loaded:
            return _model is not None
        path = Path(os.environ.get("CLASSIFIER_MODEL_PATH", _DEFAULT_PATH))
        if not path.exists():
            logger.warning("classifier model not found at %s; classifier tier disabled", path)
            _loaded = True
            return False
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            saved_names = list(payload["feature_names"])

            # Guard against silent feature-schema drift: if app/features.py has
            # been edited since the model was trained (features added, removed,
            # renamed, or reordered), the saved model would receive misaligned
            # inputs and produce garbage. Disable the classifier tier loudly
            # rather than failing silently — the caller falls back to the
            # heuristic+LLM path which handles this gracefully.
            from app.features import feature_names as _live_feature_names
            live_names = _live_feature_names()
            if saved_names != live_names:
                logger.error(
                    "feature schema drift detected: saved model expects %d "
                    "features, current featurize() produces %d. "
                    "First mismatch at index %d: saved=%r vs current=%r. "
                    "Classifier tier disabled — retrain via "
                    "`python -m eval.train_classifier --save`.",
                    len(saved_names), len(live_names),
                    next((i for i, (a, b) in enumerate(zip(saved_names, live_names, strict=False)) if a != b), -1),
                    saved_names[: len(live_names)],
                    live_names[: len(saved_names)],
                )
                _loaded = True
                return False

            _model = payload["model"]
            _feature_names = saved_names
            _threshold = float(os.environ.get("CLASSIFIER_THRESHOLD", payload.get("threshold", 0.5)))
            logger.info(
                "classifier model loaded: %d features, threshold=%.2f",
                len(_feature_names), _threshold,
            )
        except Exception as e:
            logger.warning("failed to load classifier model from %s: %s", path, e)
            _model = None
        _loaded = True
        return _model is not None


def is_available() -> bool:
    return _load_once()


def predict_batch(
    pairs: list[tuple[str, str, str | None, str | None, StudyTags, StudyTags]],
) -> list[bool] | None:
    """Predict for a batch of (curr_desc, prior_desc, curr_date, prior_date, curr_tags, prior_tags).

    Returns None if the model is not loaded; otherwise returns one bool per pair.
    """
    if not _load_once():
        return None
    if not pairs:
        return []

    assert _model is not None and _feature_names is not None  # noqa: S101  # invariant after _load_once() returns True

    n = len(pairs)
    X = np.zeros((n, len(_feature_names)), dtype=np.float32)
    for i, (cd, pd, cdate, pdate, ct, pt) in enumerate(pairs):
        fb = featurize(cd, pd, cdate, pdate, ct, pt)
        X[i, :] = fb.values

    proba = _model.predict(X)
    return [bool(p >= _threshold) for p in proba]
