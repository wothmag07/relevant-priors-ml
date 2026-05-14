"""Parse radiology study_description strings into structured tags.

Strategy: regex-based keyword matching on a normalized form of the description.
Each description maps to a set of *region* tags plus optional modality/contrast/laterality.
Relevance is then determined primarily by region-set overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Optional


def _norm(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 /_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Region keyword → canonical region tag.
# Order matters where prefixes overlap; longer/more-specific entries first.
REGION_PATTERNS: list[tuple[str, str]] = [
    # "HEAD AND NECK" / "HEAD/NECK" — almost always soft-tissue neck, not brain.
    # Must come before the bare HEAD pattern. Note: \W+ alone wouldn't catch
    # the literal " AND " separator (A/N/D are word chars), so we match it
    # explicitly.
    (r"\bHEAD\s+AND\s+NECK\b|\bHEAD\W+NECK\b|\bH/N\b|\bHEAD/NECK\b", "neck"),
    # Bone scan / whole-body NM imaging
    (r"\bBONE SCAN\b|\bSKELETAL SURVEY\b", "wholebody"),
    # Ultrasound breast screening variants that don't say "breast" or "mam"
    (r"\bULTRASOUND BILAT SCREEN\b", "breast"),
    # Vascular (must come before generic anatomy because "CAROTID" has its own meaning)
    (r"\bCAROTID\b", "vasc_carotid"),
    (r"\bTRANSCRANIAL\b", "vasc_carotid"),
    (r"\bVENOUS\b.*\b(LEG|LEGS|LE)\b", "vasc_le"),
    (r"\bVAS\b.*\b(LE|LEG)\b", "vasc_le"),
    (r"\bDOPPL?ER?\b.*\b(LEG|LE)\b", "vasc_le"),
    (r"\b(UE|UPPER EXTREM|UP VENOUS|ARM)\b", "vasc_ue"),
    (r"\bAORTA\b", "vasc_aorta"),
    (r"\bRENAL ART\b", "vasc_renal"),
    # Cardiac
    (r"\bECHO\b", "heart"),
    (r"\bTTE\b", "heart"),
    (r"\bN?M\s*MYO\s*PERF\b|\bMYO ?PERF\b", "heart"),
    (r"\bMYOCARD\b", "heart"),
    (r"\bSPECT\b", "heart"),  # in this dataset SPECT is myocardial perfusion
    (r"\bCORONARY\b", "heart"),
    (r"\bCARDIAC\b", "heart"),
    # Breast / mammography
    (r"\bMAM\b|\bMAMMO\w*\b", "breast"),
    (r"\bBREAST\b", "breast"),
    # Brain / head / skull
    (r"\bBRAIN\b", "brain"),
    (r"\bHEAD\b", "brain"),
    (r"\bSKULL\b(?! TO )", "brain"),  # "skull to thigh" is whole-body PET, not skull
    (r"\bCEREBRAL\b", "brain"),
    # Sinuses / maxillofacial
    (r"\bSINUS\w*\b", "sinuses"),
    (r"\bMAXFACIAL\b|\bMAXILLOFACIAL\b|\bFACIAL\b", "sinuses"),
    (r"\bORBIT\w*\b", "sinuses"),
    # Neck / thyroid / soft tissue neck
    (r"\bTHYROID\b", "neck"),
    (r"\bSOFT TISSUE NECK\b", "neck"),
    (r"\bNECK\b", "neck"),
    # Spine
    (r"\bC[ -]?SPINE\b|\bCERVICAL SPINE\b|\bCERVICL SPINE\b|\bCERV SPINE\b", "c_spine"),
    (r"\bT[ -]?SPINE\b|\bTHORACIC SPINE\b|\bTHOR SPINE\b", "t_spine"),
    (r"\bL[ -]?SPINE\b|\bLUMBAR SPINE\b|\bLUMBAR\b|\bLUM SPINE\b|\bSPINE\W*LUMBAR\b", "l_spine"),
    (r"\bSACRUM\b|\bSACRAL\b|\bCOCCYX\b", "sacrum"),
    (r"\bSPINE\b", "spine_other"),
    # Chest / lungs
    (r"\bCHEST\b", "chest"),
    (r"\bLUNG\w*\b", "chest"),
    (r"\bTHORAX\b", "chest"),
    (r"\bRIB\w*\b", "chest"),
    # Abdomen and pelvis (compound first). Note: \W in Python regex does NOT
    # match underscore (since _ is a word char), so we use an explicit class
    # to handle "ABD_PEL", "ABD/PEL", "ABD PEL".
    (r"\bABD(?:OMEN)?[ /_\-]+PEL\w*\b|\bABD AND PEL\b", "abd_pel"),
    (r"\bABDOMEN\b|\bABD\b|\bABDOMINAL\b", "abdomen"),
    (r"\bKUB\b", "abdomen"),
    (r"\bRENAL COLIC\b", "abd_pel"),
    (r"\bPELVIS\b|\bPELVIC\b", "pelvis"),
    (r"\bENDOVAGINAL\b|\bTRANSVAGINAL\b|\bUTERUS\b|\bOVAR\w*\b", "pelvis"),
    (r"\bKIDNEY\w*\b|\bRENAL\b", "abdomen"),
    (r"\bLIVER\b|\bHEPAT\w*\b|\bGALLBLAD\w*\b|\bBILIARY\b", "abdomen"),
    # GI fluoro
    (r"\bESOPHAG\w*\b|\bBARIUM\b|\bGI SERIES\b|\bUPPER GI\b", "gi_fluoro"),
    # Whole-body PET
    (r"\bSKULL TO THIGH\b|\bWHOLE BODY\b", "wholebody"),
    # Bone density
    (r"\bDXA\b|\bBONE DENS\w*\b", "bone_density"),
    # Joints / extremities
    (r"\bSHOULDER\b", "shoulder"),
    (r"\bHIP\b", "hip"),
    (r"\bKNEE\b", "knee"),
    (r"\bANKLE\b", "ankle"),
    (r"\bFOOT\b|\bFEET\b|\bTOE\w*\b", "foot"),
    (r"\bELBOW\b", "elbow"),
    (r"\bWRIST\b", "wrist"),
    (r"\bHAND\b|\bFINGER\w*\b", "hand"),
    (r"\bFEMUR\b", "femur"),
    (r"\bTIBIA\b|\bFIBULA\b", "tib_fib"),
    (r"\bHUMERUS\b", "humerus"),
    (r"\bCLAVICLE\b", "clavicle"),
    # EEG and neuro physiology
    (r"\bEEG\b", "eeg"),
]

# Modality detection. Some modality words also imply region (ECHO->heart, DXA->bone_density)
# but we still record the modality separately.
MODALITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bCTA\b|\bCT ANGIO\w*\b", "cta"),
    (r"\bMRA\b|\bMR ANGIO\w*\b", "mra"),
    (r"\bMRI\b|\bMR\b(?! ANGIO)", "mri"),
    (r"\bCT\b", "ct"),
    (r"\bMAMMO\w*\b|\bMAM\b", "mammo"),
    (r"\bUS\b|\bULTRASOUND\b|\bSONOGR\w*\b|\bECHO\b|\bDOPPL?ER?\b", "us"),
    (r"\bPET\b", "pet"),
    (r"\bSPECT\b|\bMYO PERF\b|\bNM\b|\bNUCLEAR\b", "nm"),
    (r"\bDXA\b|\bBONE DENS\w*\b", "dxa"),
    (r"\bEEG\b", "eeg"),
    (r"\bFLUORO\w*\b|\bBARIUM\b|\bESOPHAG\w*\b|\bGI SERIES\b", "fluoro"),
    # XR / plain film
    (r"\bXR\b|\bX-?RAY\b|\bRADIOGRAPH\w*\b", "xr"),
    (r"\b\d+\s*VIEW", "xr"),
    (r"\bAP\b|\bPA\b|\bLAT\b|\bFRONTAL\b", "xr"),
]

# Plain-word descriptors (e.g. "Chest", "Abdomen", "Breast", "Thyroid") - these are
# implicitly XR/plain films of that region in this dataset (frequent in the data).
PLAIN_REGION_FALLBACK = {
    "CHEST": ("chest", "xr"),
    "ABDOMEN": ("abdomen", "xr"),
    "PELVIC": ("pelvis", "xr"),
    "BREAST": ("breast", "mammo"),
    "THYROID": ("neck", "us"),
    "BONE DENSITY": ("bone_density", "dxa"),
}


@dataclass(frozen=True)
class StudyTags:
    regions: FrozenSet[str]
    modality: Optional[str]
    contrast: Optional[str]  # 'with' | 'without' | 'with_without' | None
    laterality: Optional[str]  # 'left' | 'right' | 'bilateral' | None
    is_outside: bool = False
    raw_norm: str = ""


def _detect_contrast(s: str) -> Optional[str]:
    has_with = bool(re.search(r"\bW\b|\bWITH\b|\bW/\b|\bW CON\b|\bWITH CON\w*\b|\bWITH CNTR\w*\b|\bW CNTR\w*\b", s))
    has_without = bool(re.search(r"\bWO\b|\bWITHOUT\b|\bW/O\b|\bWO CON\b|\bWITHOUT CON\w*\b|\bWITHOUT CNTR\w*\b|\bWO CNTR\w*\b", s))
    # combined like "wo/w" or "WITHOUT/WITH"
    if re.search(r"\bWO/W\b|\bW/WO\b|\bWITHOUT/WITH\b|\bWITH/WITHOUT\b|\bWO\s*W\b", s):
        return "with_without"
    if has_with and has_without:
        return "with_without"
    if has_with:
        return "with"
    if has_without:
        return "without"
    return None


def _detect_laterality(s: str) -> Optional[str]:
    if re.search(r"\bBI\b|\bBIL\b|\bBILAT\w*\b|\bBOTH\b", s):
        return "bilateral"
    has_left = bool(re.search(r"\bLEFT\b|\bLT\b|\bL\b(?! SPINE)", s))
    has_right = bool(re.search(r"\bRIGHT\b|\bRT\b|\bR\b(?! SPINE)", s))
    if has_left and has_right:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


# Tags that, when present, override all other region tags. e.g. DXA hip/spine
# imaging is its own category — it's only relevant to other DXA studies in this
# dataset, not to MRI hip or spine X-ray.
EXCLUSIVE_REGION_TAGS = {"bone_density", "eeg"}

# When a more-specific spine tag matches, drop the generic spine_other.
SPECIFIC_SPINE_TAGS = {"c_spine", "t_spine", "l_spine", "sacrum"}


def parse_description(description: str) -> StudyTags:
    s = _norm(description)
    if not s:
        return StudyTags(frozenset(), None, None, None, False, "")

    # Special: "outside films" - radiologists almost always look at outside priors
    if "OUTSIDE FILMS" in s or s == "OUTSIDE":
        return StudyTags(frozenset({"unknown"}), None, None, None, True, s)

    regions: set[str] = set()
    matched_spans: list[tuple[int, int]] = []
    for pat, tag in REGION_PATTERNS:
        for m in re.finditer(pat, s):
            span = (m.start(), m.end())
            # Honour the priority-by-order contract: skip this match if it
            # overlaps a span already claimed by a higher-priority pattern.
            # Without this, "HEAD AND NECK" tags as both 'neck' (correct) AND
            # 'brain' (the bare HEAD pattern firing inside the same span).
            if any(span[0] < pe and ps < span[1] for ps, pe in matched_spans):
                continue
            regions.add(tag)
            matched_spans.append(span)

    # Apply exclusive-tag override
    exclusive_present = regions & EXCLUSIVE_REGION_TAGS
    if exclusive_present:
        regions = exclusive_present

    # Drop generic spine when a specific spine level is present
    if regions & SPECIFIC_SPINE_TAGS:
        regions.discard("spine_other")

    if not regions:
        # Plain-word fallbacks (e.g. "Chest", "Abdomen") — only when description is short
        # and contains exactly that word.
        for word, (region, _modality) in PLAIN_REGION_FALLBACK.items():
            if word in s and len(s) <= len(word) + 4:
                regions.add(region)

    modality: Optional[str] = None
    for pat, mod in MODALITY_PATTERNS:
        if re.search(pat, s):
            modality = mod
            break

    if modality is None:
        for word, (_region, mod) in PLAIN_REGION_FALLBACK.items():
            if word in s and len(s) <= len(word) + 4:
                modality = mod
                break

    contrast = _detect_contrast(s)
    laterality = _detect_laterality(s)

    return StudyTags(
        regions=frozenset(regions) if regions else frozenset({"unknown"}),
        modality=modality,
        contrast=contrast,
        laterality=laterality,
        is_outside=False,
        raw_norm=s,
    )
