"""Unit tests for parser and heuristic edge cases.

Each rule below is called out in `experiments.md` as either a deliberate
inclusion (e.g. `OUTSIDE FILMS → False`, `bone_density` exclusive) or a
deliberate exclusion (e.g. carotid does not match brain, EEG does not match
brain). These tests pin those decisions so a future regex tweak can't
silently regress them.
"""
from __future__ import annotations

from app.heuristic import _regions_overlap, classify_pair
from app.parser import parse_description

# ---------------------------------------------------------------------------
# Region tagging
# ---------------------------------------------------------------------------

def test_abd_pel_underscore_form():
    """The original parser bug: \\W excludes _ in Python regex, so 'ABD_PEL'
    used to fall through. Pinned to ensure the explicit [_/-] class stays."""
    tags = parse_description("CT ABD_PEL WITHOUT CONTRAST")
    assert "abd_pel" in tags.regions


def test_abd_pel_slash_form():
    tags = parse_description("CT ABD/PEL WITH CONTRAST")
    assert "abd_pel" in tags.regions


def test_abdomen_and_pelvis_full_spelling():
    """'CT ABDOMEN AND PELVIS' tags as separate {abdomen, pelvis} rather than
    the combined abd_pel tag. That's fine: it still overlaps with any abd_pel
    description because abd_pel expands to {abdomen, pelvis}."""
    tags = parse_description("CT ABDOMEN AND PELVIS WITHOUT CONTRAST")
    assert "abdomen" in tags.regions
    assert "pelvis" in tags.regions
    # And it must overlap with a CT abd/pel prior:
    other = parse_description("CT ABD/PEL WITH CONTRAST")
    assert _regions_overlap(tags.regions, other.regions) is True


def test_outside_films_flagged():
    """OUTSIDE FILMS is uniformly negative in the public split (0/67); the
    parser flags is_outside=True so the heuristic can return False with
    high confidence."""
    tags = parse_description("OUTSIDE FILMS")
    assert tags.is_outside is True
    assert "unknown" in tags.regions


def test_outside_films_does_not_match_outside_screening():
    """'OUTSIDE SCREENING US BREAST BILATERAL' is a real breast study that
    happens to be done at an outside facility — must NOT trip is_outside."""
    tags = parse_description("OUTSIDE SCREENING US BREAST BILATERAL")
    assert tags.is_outside is False
    assert "breast" in tags.regions


def test_head_and_neck_does_not_collapse_to_brain():
    """'HEAD AND NECK CT' is a soft-tissue neck study, not brain.
    Regression test for the priority ordering of REGION_PATTERNS."""
    tags = parse_description("CT HEAD AND NECK WITH CONTRAST")
    assert "neck" in tags.regions
    assert "brain" not in tags.regions


def test_bare_head_maps_to_brain():
    tags = parse_description("CT HEAD WITHOUT CONTRAST")
    assert "brain" in tags.regions


def test_bone_density_is_exclusive():
    """DXA hip studies should NOT also tag as hip — bone_density is an
    exclusive override (DXA is only relevant to other DXA in this dataset)."""
    tags = parse_description("BONE DENSITY HIP")
    assert tags.regions == frozenset({"bone_density"})


def test_eeg_is_exclusive():
    """EEG studies should NOT additionally tag as brain even if the
    description mentions HEAD."""
    tags = parse_description("EEG HEAD STANDARD")
    assert tags.regions == frozenset({"eeg"})


def test_skull_to_thigh_is_wholebody_not_skull():
    """'PET-CT SKULL TO THIGH' is a whole-body PET — must not tag as brain."""
    tags = parse_description("PET-CT SKULL TO THIGH SUBSQNT")
    assert "wholebody" in tags.regions
    assert "brain" not in tags.regions


# ---------------------------------------------------------------------------
# Modality / contrast / laterality
# ---------------------------------------------------------------------------

def test_laterality_left():
    assert parse_description("KNEE, LEFT - 1 OR 2 VIEWS").laterality == "left"


def test_laterality_right():
    assert parse_description("XR forearm RT").laterality == "right"


def test_laterality_bilateral():
    assert parse_description("MAMMOGRAM SCREENING BILATERAL").laterality == "bilateral"


def test_modality_ct():
    assert parse_description("CT CHEST WITHOUT CNTRST").modality == "ct"


def test_modality_mri_not_mra():
    """MRI must not collapse to MRA — the 'MR' pattern explicitly negates ANGIO."""
    assert parse_description("MRI BRAIN STROKE LIMITED").modality == "mri"


def test_modality_cta_takes_precedence_over_ct():
    assert parse_description("CTA HEAD WITH CONTRAST").modality == "cta"


def test_contrast_with_without():
    assert parse_description("MRI brain wo/w contrast").contrast == "with_without"


# ---------------------------------------------------------------------------
# classify_pair / _regions_overlap — data-validated cross-region exclusions
# ---------------------------------------------------------------------------

def test_carotid_does_not_match_brain():
    """In the public split: 65 / 141 pos/neg → blanket link rejected
    because labelers consider these clinically separate."""
    curr = parse_description("CTA HEAD WITH CONTRAST")          # → {brain}
    prior = parse_description("CTA CAROTID")                     # → {vasc_carotid}
    assert _regions_overlap(curr.regions, prior.regions) is False


def test_eeg_does_not_match_brain_mri():
    """EEG ↔ brain MRI: 30/32 in the public split, barely net-negative.
    Pinned because EEG is exclusive and intentionally doesn't bridge to brain."""
    curr = parse_description("MRI BRAIN WITHOUT CONTRAST")
    prior = parse_description("EEG STANDARD")
    assert _regions_overlap(curr.regions, prior.regions) is False


def test_heart_does_not_match_chest():
    """heart ↔ chest: 158/506 — strongly net-negative, kept disjoint."""
    curr = parse_description("ECHO 2D Mmode transthorac TTE")    # → {heart}
    prior = parse_description("XR chest 2V PA/lat")               # → {chest}
    assert _regions_overlap(curr.regions, prior.regions) is False


def test_t_spine_does_not_match_chest():
    """t_spine ↔ chest: 49/99 — kept disjoint."""
    curr = parse_description("MRI T-SPINE WITHOUT CONTRAST")
    prior = parse_description("CT CHEST WITH CNTRST")
    assert _regions_overlap(curr.regions, prior.regions) is False


def test_l_spine_does_not_match_abdomen():
    """Newly-tested (and rejected) cross-region: l_spine ↔ abdomen 37/208."""
    curr = parse_description("MRI L-SPINE WITHOUT CONTRAST")
    prior = parse_description("CT ABDOMEN WITHOUT CONTRAST")
    assert _regions_overlap(curr.regions, prior.regions) is False


# ---------------------------------------------------------------------------
# Coverage expansions — regions that DO overlap
# ---------------------------------------------------------------------------

def test_abd_pel_overlaps_abdomen():
    """abd_pel expands to {abdomen, pelvis} — a CT abdomen prior is comparable
    to a CT abd/pel current."""
    curr = parse_description("CT ABDOMEN AND PELVIS WITHOUT CONTRAST")
    prior = parse_description("CT ABDOMEN WITH CNTRST")
    assert _regions_overlap(curr.regions, prior.regions) is True


def test_wholebody_overlaps_chest():
    """wholebody expands to torso regions — bone scan/PET should match chest CT."""
    curr = parse_description("CT CHEST WITHOUT CONTRAST")
    prior = parse_description("PET-CT SKULL TO THIGH SUBSQNT")
    assert _regions_overlap(curr.regions, prior.regions) is True


# ---------------------------------------------------------------------------
# classify_pair — confidence + reason wiring
# ---------------------------------------------------------------------------

def test_exact_match_high_confidence():
    """Same description on both sides → exact_match branch with conf 0.97."""
    t = parse_description("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST")
    r = classify_pair(t, t)
    assert r.predicted is True
    assert r.confidence >= 0.95
    assert r.reason == "exact_match"


def test_no_overlap_high_confidence():
    """Disjoint anatomy → no_region_overlap branch with conf 0.95."""
    knee = parse_description("KNEE, LEFT - 1 OR 2 VIEWS")
    brain = parse_description("MRI BRAIN STROKE LIMITED WITHOUT CONTRAST")
    r = classify_pair(brain, knee)
    assert r.predicted is False
    assert r.confidence >= 0.9
    assert r.reason == "no_region_overlap"


def test_outside_films_low_confidence():
    """Outside films pair returns False at modest confidence so the LLM tier
    can second-guess if needed."""
    outside = parse_description("OUTSIDE FILMS")
    brain = parse_description("MRI BRAIN")
    r = classify_pair(brain, outside)
    assert r.predicted is False
    assert r.reason == "outside_films_default_false"
