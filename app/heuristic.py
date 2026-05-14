"""Region-overlap heuristic for prior-relevance.

Returns (predicted_is_relevant, confidence) where confidence in [0,1].
The hybrid layer escalates low-confidence pairs to an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.parser import StudyTags, parse_description


# Region pairs that are *related but not identical* — both should be treated
# as overlapping for relevance purposes. Symmetric. Each entry below was
# validated against the public split: only kept where positive cases >> negative.
RELATED_REGION_PAIRS: set[frozenset[str]] = {
    # Wholebody PET / bone scan covers most of the torso → comparable.
    frozenset({"chest", "wholebody"}),
    frozenset({"abdomen", "wholebody"}),
    frozenset({"pelvis", "wholebody"}),
    frozenset({"abd_pel", "wholebody"}),
    # Note: heart↔chest, brain↔sinuses, brain↔carotid, t_spine↔chest were tested
    # and found to be net-negative on the public split — kept disjoint.
}

# When current contains all regions of prior (or vice-versa via abd_pel covers abd & pel)
COVERAGE_EXPANSIONS: dict[str, set[str]] = {
    "abd_pel": {"abdomen", "pelvis"},
    "wholebody": {"chest", "abdomen", "pelvis", "abd_pel", "heart"},
}


@dataclass(frozen=True)
class HeuristicResult:
    predicted: bool
    confidence: float  # 0..1, where 1 means "very sure"
    reason: str


def _expand_regions(regions: frozenset[str]) -> frozenset[str]:
    out = set(regions)
    for r in regions:
        if r in COVERAGE_EXPANSIONS:
            out |= COVERAGE_EXPANSIONS[r]
    return frozenset(out)


def _regions_overlap(a: frozenset[str], b: frozenset[str]) -> bool:
    if a & b:
        return True
    ae = _expand_regions(a)
    be = _expand_regions(b)
    if ae & be:
        return True
    for ra in a:
        for rb in b:
            if frozenset({ra, rb}) in RELATED_REGION_PAIRS:
                return True
    return False


def classify_pair(curr: StudyTags, prior: StudyTags) -> HeuristicResult:
    """Return relevance prediction + confidence for a (current, prior) pair."""
    # Either side unknown -> low confidence, defer to LLM in hybrid mode.
    # Note: "OUTSIDE FILMS" priors look like they'd be relevant in real radiology
    # but are uniformly labeled non-relevant in this split (67/67) — we follow
    # the data with low confidence so LLM can still override if needed.
    if "unknown" in curr.regions or "unknown" in prior.regions:
        if prior.is_outside or curr.is_outside:
            return HeuristicResult(False, 0.6, "outside_films_default_false")
        return HeuristicResult(False, 0.3, "unknown_region")

    overlap = _regions_overlap(curr.regions, prior.regions)
    if not overlap:
        # Different anatomy — strong signal that prior is not relevant.
        return HeuristicResult(False, 0.95, "no_region_overlap")

    # Regions overlap. Same exact region set + same modality = very confident True.
    same_regions = curr.regions == prior.regions
    same_modality = (curr.modality is not None and curr.modality == prior.modality)

    if same_regions and same_modality:
        return HeuristicResult(True, 0.97, "exact_match")
    if same_regions:
        return HeuristicResult(True, 0.9, "same_region_diff_modality")
    # Partial overlap (e.g. CT abd/pel vs CT abdomen): comparison still valuable.
    return HeuristicResult(True, 0.8, "partial_region_overlap")


def predict_pair(current_description: str, prior_description: str) -> HeuristicResult:
    return classify_pair(parse_description(current_description), parse_description(prior_description))
