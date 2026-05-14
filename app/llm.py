"""Batched OpenAI classifier for ambiguous (current, prior) description pairs.

Design:
* The hybrid classifier sends only pairs whose heuristic confidence is below a
  threshold. Each LLM call covers ONE current_description + many prior
  descriptions for that case (batched), since the evaluator's 360s timeout
  does not tolerate one call per prior.
* Results are cached on disk keyed by (curr_desc, prior_desc) -- dates and
  patient identity do not matter for relevance in this task. Once warm, repeat
  evaluations are essentially free.
* If the API key is missing or any LLM call fails, we degrade to the heuristic
  prediction silently -- never raise inside the request path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    predicted: bool
    source: str  # "cache" | "llm" | "fallback"


SYSTEM_PROMPT = (
    "You are a radiologist's assistant. For a given CURRENT examination, decide "
    "for each PRIOR examination whether it should be shown to the radiologist "
    "while reading the CURRENT examination.\n"
    "\n"
    "A prior is RELEVANT if it images the same or substantially overlapping "
    "anatomy (same region or covered by a wider study), regardless of modality "
    "or contrast — radiologists routinely compare across modalities.\n"
    "\n"
    "A prior is NOT relevant if it images a different anatomical region, OR if "
    "it images anatomically adjacent regions but is unlikely to add value to the "
    "current read. Labelers in this dataset are conservative — when in doubt, "
    "default to NOT relevant.\n"
    "\n"
    "Specific guidance (validated against labeled data):\n"
    "- Bone-density / DXA studies are only relevant to other DXA studies.\n"
    "- 'OUTSIDE FILMS' priors are NOT shown.\n"
    "- A wholebody PET/CT or bone scan is generally relevant to torso CT/MRI "
    "but NOT to plain-film chest X-rays.\n"
    "- Cardiac echo is generally NOT comparable to plain-film chest X-rays, "
    "but IS comparable to chest CT/MRI.\n"
    "- Spine studies (cervical, thoracic, lumbar) are generally NOT relevant to "
    "chest, abdomen, or pelvis imaging, even though anatomy is adjacent.\n"
    "- Different spine levels are generally NOT relevant to each other "
    "(cervical vs thoracic vs lumbar are read as separate problems).\n"
    "- Carotid vascular studies (CTA / MRA carotid) are generally NOT relevant "
    "to brain CT/MRI.\n"
    "- Barium swallow / GI fluoroscopy is generally NOT relevant to chest imaging.\n"
    "- EEG is its own category — generally NOT relevant to brain CT/MRI.\n"
    "\n"
    "Reply ONLY with the requested JSON object. Do not add commentary."
)

USER_TEMPLATE = (
    "CURRENT examination: {curr}\n\n"
    "PRIOR examinations (1-indexed):\n"
    "{priors}\n\n"
    "Return JSON exactly like:\n"
    '{{"predictions": [{{"i": 1, "relevant": true}}, {{"i": 2, "relevant": false}}, ...]}}'
)


class _LLMCache:
    """Thread-safe disk-backed cache. JSON for human-readable diffing."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._dirty = False
        self._data: dict[str, bool] = {}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:  # corrupt file shouldn't kill startup
                logger.warning("could not load cache %s: %s", path, e)

    @staticmethod
    def _key(curr: str, prior: str) -> str:
        return curr.strip() + " || " + prior.strip()

    def get(self, curr: str, prior: str) -> Optional[bool]:
        return self._data.get(self._key(curr, prior))

    def set_many(self, items: Iterable[tuple[str, str, bool]]) -> None:
        with self._lock:
            for curr, prior, val in items:
                self._data[self._key(curr, prior)] = val
                self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=0, sort_keys=True)
            tmp.replace(self.path)
            self._dirty = False

    def __len__(self) -> int:
        return len(self._data)


_cache_singleton: Optional[_LLMCache] = None


def get_cache() -> _LLMCache:
    global _cache_singleton
    if _cache_singleton is None:
        path = Path(os.environ.get("CACHE_PATH", ".cache/llm_cache.json"))
        _cache_singleton = _LLMCache(path)
    return _cache_singleton


def llm_enabled() -> bool:
    return os.environ.get("LLM_ENABLED", "1") != "0" and bool(os.environ.get("OPENAI_API_KEY"))


async def _call_openai(curr: str, priors: list[str], model: str) -> list[bool]:
    """One batched LLM call for one current + N priors. Returns one bool per prior."""
    from openai import AsyncOpenAI

    # Bound each call so a hung connection cannot eat the 360s evaluator budget.
    # The OpenAI SDK retries 429/5xx automatically (max_retries=2); this is the
    # outer wall-clock cap on a single attempt.
    timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "60"))
    client = AsyncOpenAI(timeout=timeout_s)
    user = USER_TEMPLATE.format(
        curr=curr,
        priors="\n".join(f"{i + 1}. {p}" for i, p in enumerate(priors)),
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=64 + 24 * len(priors),
    )
    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(content)
    out = [False] * len(priors)
    for entry in parsed.get("predictions", []):
        try:
            i = int(entry["i"]) - 1
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= i < len(priors):
            out[i] = bool(entry.get("relevant", False))
    return out


async def classify_batch_async(
    curr: str,
    priors: list[str],
    *,
    model: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> list[LLMResult]:
    """Classify (curr, priors) using cache + LLM. Always returns one result per prior."""
    cache = get_cache()
    results: list[Optional[LLMResult]] = [None] * len(priors)
    uncached_idxs: list[int] = []
    uncached_priors: list[str] = []
    for i, p in enumerate(priors):
        cached = cache.get(curr, p)
        if cached is not None:
            results[i] = LLMResult(cached, "cache")
        else:
            uncached_idxs.append(i)
            uncached_priors.append(p)

    if uncached_priors and llm_enabled():
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        try:
            if semaphore is not None:
                async with semaphore:
                    preds = await _call_openai(curr, uncached_priors, model)
            else:
                preds = await _call_openai(curr, uncached_priors, model)
            cache.set_many(
                (curr, uncached_priors[k], preds[k]) for k in range(len(uncached_priors))
            )
            for k, idx in enumerate(uncached_idxs):
                results[idx] = LLMResult(preds[k], "llm")
        except Exception as e:
            logger.warning("LLM call failed (curr=%r, n_priors=%d): %s", curr, len(uncached_priors), e)
            for idx in uncached_idxs:
                results[idx] = LLMResult(False, "fallback")  # caller will replace with heuristic

    # any still-None means LLM disabled and not cached -> caller handles fallback
    out: list[LLMResult] = []
    for r in results:
        out.append(r if r is not None else LLMResult(False, "fallback"))
    return out
