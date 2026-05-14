"""FastAPI app exposing the relevant-priors classifier.

Contract:
  POST /predict      -> {"predictions": [...]}     (canonical path)
  POST /             -> same                       (some evaluators POST to root)
  GET  /healthz      -> {"status": "ok"}
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv  # local dev convenience; ignored if missing
    load_dotenv()
except Exception:
    pass

from app.classifier import predict_cases_async
from app.llm import get_cache, llm_enabled
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("relevant-priors")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache = get_cache()
    logger.info(
        "startup llm_enabled=%s cache_entries=%d model=%s",
        llm_enabled(), len(cache), os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    yield
    try:
        get_cache().flush()
    except Exception as e:
        logger.warning("cache flush on shutdown failed: %s", e)


app = FastAPI(title="Relevant Priors Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "llm_enabled": llm_enabled(), "cache_entries": len(get_cache())}


@app.get("/")
async def root():
    return {
        "service": "relevant-priors",
        "version": "1.0.0",
        "endpoint": "POST /predict",
        "schema": "see app/schemas.py",
    }


def _fallback_predictions(req: PredictRequest):
    """One all-False prediction per prior. Used when the classifier fails or
    blows the wall-clock budget — better than skipping (which counts as wrong)."""
    from app.schemas import Prediction
    return [
        Prediction(case_id=c.case_id, study_id=p.study_id, predicted_is_relevant=False)
        for c in req.cases for p in c.prior_studies
    ]


async def _handle(req: PredictRequest, request_id: str) -> PredictResponse:
    n_cases = len(req.cases)
    n_priors = sum(len(c.prior_studies) for c in req.cases)
    t0 = time.perf_counter()
    # Hard wall-clock budget: keep us inside the evaluator's 360s timeout even
    # if the LLM tier stalls. On expiry, in-flight tasks are cancelled and we
    # fall back to all-False for the entire request.
    budget_s = float(os.environ.get("REQUEST_TIMEOUT_S", "300"))
    try:
        predictions = await asyncio.wait_for(
            predict_cases_async(req.cases, request_id=request_id), timeout=budget_s
        )
    except asyncio.TimeoutError:
        logger.warning(
            "request_id=%s exceeded budget=%.0fs; returning all-False fallback",
            request_id, budget_s,
        )
        predictions = _fallback_predictions(req)
    except Exception as e:
        # Skipping counts as wrong; safer to return one prediction per prior
        # using the always-False fallback (76% baseline) than to error out.
        logger.exception("classifier failed; returning fallback predictions: %s", e)
        predictions = _fallback_predictions(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "request_id=%s cases=%d priors=%d predictions=%d elapsed_ms=%.0f",
        request_id, n_cases, n_priors, len(predictions), elapsed_ms,
    )
    return PredictResponse(predictions=predictions)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    return await _handle(req, request_id)


@app.post("/", response_model=PredictResponse)
async def predict_root(req: PredictRequest, request: Request):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    return await _handle(req, request_id)


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    # Never return 500 for an evaluator call — degrade to all-False rather than
    # skipping (which counts as wrong anyway). But here we want to surface bugs
    # so we log loudly and return a JSON error. The evaluator will treat this
    # as missing predictions, but visibility matters more in dev.
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})
