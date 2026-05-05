"""
service_classifier.py
4-tier service classification engine.

Given a SensorEvent (asset_type + alert_type + context), returns the best
matching IFM service classification using the same cascading philosophy as
the asset matcher:

  Tier 1  Perfect  – exact asset_type + alert_type match in catalog     (≥85%)
  Tier 2  Partial  – fuzzy keyword / category match                     (50–84%)
  Tier 3  LLM      – delegate to LLM agent when fuzzy score is too low  (<50% but plausible)
  Tier 4  No Match – nothing found                                       (0%)

Result dict mirrors the work-order schema so it can be handed directly
to WorkOrderCreator.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ── Load catalog ───────────────────────────────────────────────────────────────

_CATALOG_PATH = Path(__file__).parent.parent / "data" / "service_catalog.json"

def _load_catalog() -> List[dict]:
    try:
        with open(_CATALOG_PATH, "r") as f:
            data = json.load(f)
        return data["service_classifications"]
    except Exception as exc:
        logger.error("Failed to load service catalog: %s", exc)
        return []

_CATALOG: List[dict] = _load_catalog()


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a.lower(), b.lower())


def _best_fuzzy(target: str, options: List[str]) -> float:
    if not options:
        return 0.0
    return max(_fuzzy(target, o) for o in options)


def _score_classification(
    asset_type: str,
    alert_type: str,
    description: str,
    sc: dict,
) -> float:
    """
    Score a service classification entry against a sensor event.
    Returns 0–100.
    """
    asset_score = _best_fuzzy(asset_type, sc.get("asset_types", []))
    alert_score = _best_fuzzy(alert_type, sc.get("alert_types", []))

    # Keyword boost from description
    kw_score = 0.0
    if description:
        kw_score = _best_fuzzy(description, sc.get("keywords", []))

    # Weighted composite: asset type matters most
    composite = (asset_score * 0.45) + (alert_score * 0.40) + (kw_score * 0.15)
    return round(composite, 2)


# ── Classification result ──────────────────────────────────────────────────────

def _build_result(
    sc: dict,
    confidence: float,
    match_type: str,
    reasoning: str = "",
    event_id: str = "",
    asset_id: str = "",
    location_id: str = "",
) -> dict:
    return {
        "event_id": event_id,
        "asset_id": asset_id,
        "location_id": location_id,
        "service_classification_id": sc["id"],
        "service_classification_name": sc["name"],
        "category": sc["category"],
        "subcategory": sc["subcategory"],
        "priority": sc["priority"],
        "sla_hours": sc["sla_hours"],
        "confidence": round(confidence, 1),
        "match_type": match_type,
        "reasoning": reasoning,
        "auto_create_threshold": sc.get("auto_create_threshold", 85),
    }


# ── Main classifier ────────────────────────────────────────────────────────────

class ServiceClassifier:
    """
    4-tier cascading service classification engine.

    Usage:
        classifier = ServiceClassifier()
        result = classifier.classify(event)
    """

    PERFECT_THRESHOLD = 85.0
    PARTIAL_THRESHOLD = 50.0
    LLM_THRESHOLD     = 30.0   # below this → No Match without LLM

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._catalog = _CATALOG

    def classify(self, event_or_dict) -> dict:
        """
        Main entry point.  Accepts a SensorEvent or a plain dict
        with keys: asset_type, alert_type, description (optional),
        event_id, asset_id, location_id.
        """
        if hasattr(event_or_dict, "asset_type"):
            asset_type   = event_or_dict.asset_type
            alert_type   = event_or_dict.alert_type
            description  = getattr(event_or_dict, "asset_name", "")
            event_id     = event_or_dict.event_id
            asset_id     = event_or_dict.asset_id
            location_id  = event_or_dict.location_id
        else:
            asset_type   = str(event_or_dict.get("asset_type", ""))
            alert_type   = str(event_or_dict.get("alert_type", ""))
            description  = str(event_or_dict.get("description", ""))
            event_id     = str(event_or_dict.get("event_id", ""))
            asset_id     = str(event_or_dict.get("asset_id", ""))
            location_id  = str(event_or_dict.get("location_id", ""))

        # ── Tier 1: Perfect (exact/high-confidence) ───────────────────────────
        scores = [
            (_score_classification(asset_type, alert_type, description, sc), sc)
            for sc in self._catalog
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sc = scores[0] if scores else (0, None)

        if best_sc and best_score >= self.PERFECT_THRESHOLD:
            return _build_result(
                best_sc, best_score, "Perfect Match",
                reasoning="Exact/high-confidence catalog match.",
                event_id=event_id, asset_id=asset_id, location_id=location_id,
            )

        # ── Tier 2: Partial (fuzzy) ────────────────────────────────────────────
        if best_sc and best_score >= self.PARTIAL_THRESHOLD:
            return _build_result(
                best_sc, best_score, "Partial Match",
                reasoning=f"Fuzzy match at {best_score:.1f}% confidence. Human review recommended.",
                event_id=event_id, asset_id=asset_id, location_id=location_id,
            )

        # ── Tier 3: LLM Reasoning ─────────────────────────────────────────────
        if self.use_llm and best_sc and best_score >= self.LLM_THRESHOLD:
            try:
                result = self._llm_classify(
                    asset_type, alert_type, description,
                    scores[:5], event_id, asset_id, location_id,
                )
                if result:
                    return result
            except Exception as exc:
                logger.warning("LLM classification failed: %s", exc)

        # ── Tier 4: No Match ──────────────────────────────────────────────────
        return {
            "event_id": event_id,
            "asset_id": asset_id,
            "location_id": location_id,
            "service_classification_id": None,
            "service_classification_name": "No Match",
            "category": None,
            "subcategory": None,
            "priority": None,
            "sla_hours": None,
            "confidence": 0.0,
            "match_type": "No Match",
            "reasoning": "No suitable service classification found.",
            "auto_create_threshold": 85,
        }

    def classify_batch(self, events: list) -> List[dict]:
        return [self.classify(e) for e in events]

    # ── LLM fallback ──────────────────────────────────────────────────────────

    def _llm_classify(
        self,
        asset_type: str,
        alert_type: str,
        description: str,
        top_candidates: List[Tuple[float, dict]],
        event_id: str,
        asset_id: str,
        location_id: str,
    ) -> Optional[dict]:
        """
        Ask LLM to pick the best service classification from top candidates.
        Returns a result dict or None if LLM can't decide.
        """
        from llm.service_classification_agent import classify_with_llm
        result_sc, confidence, reasoning = classify_with_llm(
            asset_type=asset_type,
            alert_type=alert_type,
            description=description,
            candidates=[{"id": sc["id"], "name": sc["name"], "score": score}
                        for score, sc in top_candidates],
        )
        if result_sc:
            return _build_result(
                result_sc, confidence, "LLM Reasoned",
                reasoning=reasoning,
                event_id=event_id, asset_id=asset_id, location_id=location_id,
            )
        return None
