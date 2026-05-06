"""
matcher.py
Multi-signal fuzzy scoring engine that implements the 5 approaches
defined in the SFM-IFM mapping PPTX.

Approach hierarchy (highest confidence first):
  Perfect-1  – Name + Make/Model/Serial + Location  (no building)
  Perfect-2  – Name + Location + Building            (no make/model/serial)
  Perfect-3  – Make/Model/Serial + Location + Building (no name)
  Partial-1  – Name (50 %) + Location + Building
  Partial-2  – Name (50 %) + Make/Model/Serial + Location + Building
"""

from __future__ import annotations
from rapidfuzz import fuzz
from typing import Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def _s(val) -> str:
    """Safe string conversion."""
    return str(val).strip() if val else ""


def _fuzzy(a: str, b: str) -> float:
    """Token-set ratio [0-100]."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b)


def _name_score(sfm: dict, ifm: dict) -> float:
    """Best name similarity across nav_name/asset_name/position_name combos."""
    nav = _s(sfm.get("nav_name"))
    a_name = _s(ifm.get("asset_name"))
    p_name = _s(ifm.get("position_name"))
    return max(
        _fuzzy(nav, a_name),
        _fuzzy(nav, p_name),
    )


def _equip_type_ok(sfm: dict, ifm: dict) -> bool:
    """Equipment-type gate – blank SFM type always passes."""
    sfm_type = _s(sfm.get("equip_type"))
    ifm_type = _s(ifm.get("position_type_description"))
    if not sfm_type or sfm_type.lower() in ("other", "none", "null"):
        return True
    return _fuzzy(sfm_type, ifm_type) >= 85


_COUNTRY_ABBR = {
    "united states": "us", "united states of america": "us",
    "canada": "ca", "united kingdom": "uk", "great britain": "uk",
}

def _normalize_country(val: str) -> str:
    return _COUNTRY_ABBR.get(val.strip().lower(), val.strip())


def _location_score(sfm: dict, ifm: dict) -> float:
    """Country / State / City vs region_name (case-insensitive)."""
    country = _normalize_country(_s(sfm.get("country")))
    sfm_loc = ", ".join(filter(None, [country, _s(sfm.get("state")), _s(sfm.get("city"))])).lower()
    ifm_region = _s(ifm.get("region_name")).lower()
    return _fuzzy(sfm_loc, ifm_region)


def _building_score(sfm: dict, ifm: dict) -> float:
    sfm_site = _s(sfm.get("site_name") or sfm.get("building_name"))
    ifm_bldg = _s(ifm.get("building_name"))
    return _fuzzy(sfm_site, ifm_bldg)


def _make_score(sfm: dict, ifm: dict) -> float:
    return _fuzzy(_s(sfm.get("equip_make")), _s(ifm.get("manufacturer")))


def _model_score(sfm: dict, ifm: dict) -> float:
    return _fuzzy(_s(sfm.get("equip_model")), _s(ifm.get("model")))


def _serial_score(sfm: dict, ifm: dict) -> float:
    a = _s(sfm.get("equip_serial"))
    b = _s(ifm.get("serial_number"))
    if not a or not b:
        return 100.0        # treat missing serial as non-disqualifying
    return 100.0 if a == b else 0.0


# ── individual approach scorers ───────────────────────────────────────────────

def _score_approach1(sfm: dict, ifm: dict) -> Optional[float]:
    """Perfect-1: Name≥90 + Make≥95 + Model≥95 + Serial=100 + Location≥90

    Requires at least one non-blank hardware identifier (make/model/serial) in SFM.
    Records with no identifiers should fall through to Approach 2 or Partial nodes.
    """
    # Gate: SFM must carry at least one hardware identifier
    if not any(_s(sfm.get(k)) for k in ("equip_make", "equip_model", "equip_serial")):
        return None
    if not _equip_type_ok(sfm, ifm):
        return None
    ns = _name_score(sfm, ifm)
    if ns < 90:
        return None
    mk = _make_score(sfm, ifm)
    if mk < 95 and _s(sfm.get("equip_make")):
        return None
    mo = _model_score(sfm, ifm)
    if mo < 95 and _s(sfm.get("equip_model")):
        return None
    sr = _serial_score(sfm, ifm)
    if sr < 100:
        return None
    ls = _location_score(sfm, ifm)
    if ls < 85:
        return None
    return round((ns * 0.25 + mk * 0.15 + mo * 0.15 + sr * 0.10 + ls * 0.35) / 100, 4)


def _score_approach2(sfm: dict, ifm: dict) -> Optional[float]:
    """Perfect-2: EquipType + Name≥90 + Location≥85 + Building≥75"""
    if not _equip_type_ok(sfm, ifm):
        return None
    ns = _name_score(sfm, ifm)
    if ns < 90:
        return None
    ls = _location_score(sfm, ifm)
    if ls < 85:
        return None
    bs = _building_score(sfm, ifm)
    if bs < 75:
        return None
    return round((ns * 0.35 + ls * 0.35 + bs * 0.30) / 100, 4)


def _score_approach3(sfm: dict, ifm: dict) -> Optional[float]:
    """Perfect-3: EquipType + Make≥95 + Model≥95 + Serial=100 + Location≥85 + Building≥75

    Requires at least one non-blank hardware identifier in SFM.  Records with no
    identifiers fall through to Approach 2 or Partial nodes — Approach 3 must not
    become a catch-all for location+type matches.
    """
    # Gate: SFM must carry at least one hardware identifier
    if not any(_s(sfm.get(k)) for k in ("equip_make", "equip_model", "equip_serial")):
        return None
    if not _equip_type_ok(sfm, ifm):
        return None
    mk = _make_score(sfm, ifm)
    if mk < 95 and _s(sfm.get("equip_make")):
        return None
    mo = _model_score(sfm, ifm)
    if mo < 95 and _s(sfm.get("equip_model")):
        return None
    sr = _serial_score(sfm, ifm)
    if sr < 100:
        return None
    ls = _location_score(sfm, ifm)
    if ls < 85:
        return None
    bs = _building_score(sfm, ifm)
    if bs < 75:
        return None
    return round((mk * 0.20 + mo * 0.20 + ls * 0.30 + bs * 0.30) / 100, 4)


def _score_partial1(sfm: dict, ifm: dict) -> Optional[float]:
    """Partial-1: EquipType + Name≥50 + Location≥85 + Building≥50"""
    if not _equip_type_ok(sfm, ifm):
        return None
    ns = _name_score(sfm, ifm)
    if ns < 50:
        return None
    ls = _location_score(sfm, ifm)
    if ls < 85:
        return None
    bs = _building_score(sfm, ifm)
    if bs < 50:
        return None
    return round((ns * 0.35 + ls * 0.35 + bs * 0.30) / 100, 4)


def _score_partial2(sfm: dict, ifm: dict) -> Optional[float]:
    """Partial-2: EquipType + Name≥50 + Make/Model/Serial + Location≥85 + Building≥50"""
    if not _equip_type_ok(sfm, ifm):
        return None
    ns = _name_score(sfm, ifm)
    if ns < 50:
        return None
    mk = _make_score(sfm, ifm)
    mo = _model_score(sfm, ifm)
    sr = _serial_score(sfm, ifm)
    ls = _location_score(sfm, ifm)
    if ls < 85:
        return None
    bs = _building_score(sfm, ifm)
    if bs < 50:
        return None
    return round((ns * 0.20 + mk * 0.15 + mo * 0.15 + sr * 0.10 + ls * 0.20 + bs * 0.20) / 100, 4)


# ── public API ────────────────────────────────────────────────────────────────

APPROACHES = [
    ("Perfect - Approach 1", _score_approach1),
    ("Perfect - Approach 2", _score_approach2),
    ("Perfect - Approach 3", _score_approach3),
    ("Partial - Approach 1", _score_partial1),
    ("Partial - Approach 2", _score_partial2),
]


def find_best_match(sfm_record: dict, ifm_records: list[dict]) -> dict:
    """
    Run all approaches in order. Return a result dict with:
      matched_asset_id, matched_asset_name, match_type, confidence, approach_used
    """
    overall_best = {"score": 0.0, "approach": None, "ifm": None}

    for approach_name, scorer in APPROACHES:
        for ifm in ifm_records:
            score = scorer(sfm_record, ifm)
            if score is not None and score > overall_best["score"]:
                overall_best = {"score": score, "approach": approach_name, "ifm": ifm}

    if overall_best["ifm"] is None:
        return {
            "sfm_nav_name": sfm_record.get("nav_name", ""),
            "matched_asset_id": None,
            "matched_asset_name": None,
            "matched_position_name": None,
            "matched_building": None,
            "match_type": "No Match",
            "approach_used": "None",
            "confidence": 0.0,
        }

    ifm = overall_best["ifm"]
    return {
        "sfm_nav_name": sfm_record.get("nav_name", ""),
        "matched_asset_id": ifm.get("asset_id", ""),
        "matched_asset_name": ifm.get("asset_name", ""),
        "matched_position_name": ifm.get("position_name", ""),
        "matched_building": ifm.get("building_name", ""),
        "match_type": overall_best["approach"],
        "approach_used": overall_best["approach"],
        "confidence": round(overall_best["score"] * 100, 1),
    }


def run_bulk_matching(sfm_records: list[dict], ifm_records: list[dict]) -> list[dict]:
    """Match all SFM records against all IFM records. Returns list of result dicts."""
    return [find_best_match(sfm, ifm_records) for sfm in sfm_records]
