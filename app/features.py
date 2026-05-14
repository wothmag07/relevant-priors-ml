"""Feature engineering for the (current, prior) pair classifier.

Each pair is turned into a fixed-width numeric feature vector that a tree-based
model (LightGBM) can train on. Features are designed to capture:

* what the heuristic already knows (region overlap, modality match) — included
  as features so the classifier can learn when to trust or override it,
* signal the heuristic ignores (date deltas, description length, raw modality
  identities so the model can learn modality-conditional rules),
* a small one-hot expansion over the most common region tags so the model can
  pick up tag-pair-specific patterns (e.g. heart↔chest under CT/MRI but not XR).

Returns a stable feature order so a model trained once can be loaded and used
at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log1p
from typing import Optional

from app.heuristic import _expand_regions, classify_pair
from app.parser import StudyTags, parse_description


# Top region tags by occurrence in the public split. Anything outside this list
# is encoded only via the generic n_regions / overlap signals.
TOP_REGIONS = [
    "brain", "chest", "abdomen", "pelvis", "abd_pel", "breast", "heart",
    "c_spine", "t_spine", "l_spine", "knee", "hip", "shoulder",
    "wholebody", "vasc_carotid", "vasc_le", "bone_density", "neck",
    "sinuses", "eeg",
]

MODALITIES = ["ct", "mri", "xr", "mammo", "us", "pet", "nm", "dxa", "cta", "mra", "eeg", "fluoro"]

# Modalities considered "cross-sectional" — they image full anatomy in 3D
# slices, so they're more comparable to each other than to plain films.
CROSS_SECTION = {"ct", "mri", "pet", "cta", "mra"}

# Reasons emitted by classify_pair() — one-hot of which branch the heuristic took.
HEURISTIC_REASONS = [
    "exact_match",
    "same_region_diff_modality",
    "partial_region_overlap",
    "no_region_overlap",
    "unknown_region",
    "outside_films_default_false",
]


@dataclass(frozen=True)
class FeatureBundle:
    names: list[str]
    values: list[float]


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _shared_tokens(a: str, b: str) -> int:
    ta = set(a.upper().split())
    tb = set(b.upper().split())
    return len(ta & tb)


def featurize(
    curr_desc: str,
    prior_desc: str,
    curr_date: Optional[str] = None,
    prior_date: Optional[str] = None,
    curr_tags: Optional[StudyTags] = None,
    prior_tags: Optional[StudyTags] = None,
) -> FeatureBundle:
    if curr_tags is None:
        curr_tags = parse_description(curr_desc)
    if prior_tags is None:
        prior_tags = parse_description(prior_desc)

    names: list[str] = []
    values: list[float] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(float(value))

    # ── region one-hots ──
    for r in TOP_REGIONS:
        add(f"curr_{r}", r in curr_tags.regions)
        add(f"prior_{r}", r in prior_tags.regions)
        add(f"both_{r}", r in curr_tags.regions and r in prior_tags.regions)

    # ── overlap signals ──
    direct = bool(curr_tags.regions & prior_tags.regions)
    expanded = bool(_expand_regions(curr_tags.regions) & _expand_regions(prior_tags.regions))
    add("direct_overlap", direct)
    add("expanded_overlap", expanded)
    add("same_region_set", curr_tags.regions == prior_tags.regions)
    add("n_regions_curr", len(curr_tags.regions))
    add("n_regions_prior", len(prior_tags.regions))
    add("n_shared_regions", len(curr_tags.regions & prior_tags.regions))
    add("curr_unknown", "unknown" in curr_tags.regions)
    add("prior_unknown", "unknown" in prior_tags.regions)
    add("either_outside", curr_tags.is_outside or prior_tags.is_outside)

    # ── modality features ──
    for m in MODALITIES:
        add(f"curr_mod_{m}", curr_tags.modality == m)
        add(f"prior_mod_{m}", prior_tags.modality == m)
    add("same_modality", curr_tags.modality is not None
        and curr_tags.modality == prior_tags.modality)
    add("both_cross_section",
        curr_tags.modality in CROSS_SECTION and prior_tags.modality in CROSS_SECTION)
    add("either_xr", curr_tags.modality == "xr" or prior_tags.modality == "xr")
    add("modality_known_both",
        curr_tags.modality is not None and prior_tags.modality is not None)

    # ── contrast / laterality ──
    add("same_contrast", curr_tags.contrast is not None
        and curr_tags.contrast == prior_tags.contrast)
    add("with_contrast_curr", curr_tags.contrast == "with")
    add("without_contrast_curr", curr_tags.contrast == "without")
    add("same_laterality", curr_tags.laterality is not None
        and curr_tags.laterality == prior_tags.laterality)
    add("lateral_mismatch", (curr_tags.laterality == "left" and prior_tags.laterality == "right")
        or (curr_tags.laterality == "right" and prior_tags.laterality == "left"))

    # ── date delta ──
    cd = _parse_date(curr_date)
    pd = _parse_date(prior_date)
    if cd is not None and pd is not None:
        delta_days = abs((cd - pd).days)
        add("has_dates", 1.0)
        add("date_delta_days", float(delta_days))
        add("log_date_delta", log1p(delta_days))
        add("delta_under_30d", delta_days < 30)
        add("delta_under_1y", delta_days < 365)
        add("delta_under_2y", delta_days < 365 * 2)
        add("delta_over_5y", delta_days > 365 * 5)
    else:
        add("has_dates", 0.0)
        add("date_delta_days", -1.0)
        add("log_date_delta", -1.0)
        add("delta_under_30d", 0.0)
        add("delta_under_1y", 0.0)
        add("delta_under_2y", 0.0)
        add("delta_over_5y", 0.0)

    # ── description text features ──
    add("desc_len_curr", len(curr_desc))
    add("desc_len_prior", len(prior_desc))
    add("desc_len_ratio", len(curr_desc) / max(len(prior_desc), 1))
    add("shared_tokens", _shared_tokens(curr_desc, prior_desc))

    # ── heuristic signals (let the classifier learn when to trust them) ──
    h = classify_pair(curr_tags, prior_tags)
    add("heur_pred", h.predicted)
    add("heur_conf", h.confidence)
    for r in HEURISTIC_REASONS:
        add(f"heur_reason_{r}", h.reason == r)

    return FeatureBundle(names=names, values=values)


def feature_names() -> list[str]:
    """Return the canonical feature-name order without computing values."""
    return featurize("dummy", "dummy").names
