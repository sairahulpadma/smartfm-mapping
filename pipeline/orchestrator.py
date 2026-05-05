"""
orchestrator.py
End-to-end event processing pipeline.

Flow for each sensor event:
  SensorIngestor → ServiceClassifier → WorkOrderCreator/ReviewQueue

  ┌──────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
  │ Sensor Event │───▶│ Service Classifier   │───▶│ Decision Engine      │
  └──────────────┘    └─────────────────────┘    │  Perfect → IFM API   │
                                                  │  Partial → Review Q  │
                                                  │  LLM     → Review Q  │
                                                  │  No Match → Log Only │
                                                  └──────────────────────┘

The orchestrator also learns from approved/rejected reviews to improve
the training dataset (active learning loop).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from pipeline.sensor_ingestor import SensorIngestor, SensorEvent
from pipeline.service_classifier import ServiceClassifier
from pipeline.work_order_creator import WorkOrderCreator, WorkOrderDecision
from pipeline.review_queue import ReviewQueue

logger = logging.getLogger(__name__)

# ── Outcome log path ──────────────────────────────────────────────────────────
_LOG_PATH = Path(__file__).parent.parent / "data" / "pipeline_outcomes.jsonl"


class SensorToWorkOrderPipeline:
    """
    Full sensor → IFM Hub work order pipeline.

    Parameters
    ----------
    use_llm:    Enable LLM fallback in service classifier
    demo_mode:  Simulate IFM Hub API calls (no real HTTP)
    """

    def __init__(self, use_llm: bool = True, demo_mode: bool = True):
        self.ingestor   = SensorIngestor()
        self.classifier = ServiceClassifier(use_llm=use_llm)
        self.creator    = WorkOrderCreator(demo_mode=demo_mode)
        self.queue      = ReviewQueue()
        self._outcomes: List[dict] = []
        self._on_outcome: Optional[Callable] = None

        # Wire ingestor handler
        self.ingestor.register_handler(self._handle_event)

    def register_outcome_callback(self, fn: Callable[[dict], None]):
        """Register a callback invoked after each event is processed."""
        self._on_outcome = fn

    def process_raw_event(self, raw: dict) -> dict:
        """
        Main entry: ingest one raw sensor payload and run the full pipeline.
        Returns the outcome dict.
        """
        event = self.ingestor.ingest(raw)
        return self._handle_event(event)

    def process_batch(self, raw_events: List[dict]) -> List[dict]:
        """Process multiple sensor events."""
        return [self.process_raw_event(e) for e in raw_events]

    def get_outcomes(self) -> List[dict]:
        return list(self._outcomes)

    def get_summary(self) -> dict:
        outcomes = self._outcomes
        total = len(outcomes)
        if total == 0:
            return {"total": 0}

        by_decision = {}
        for o in outcomes:
            d = o.get("decision", "UNKNOWN")
            by_decision[d] = by_decision.get(d, 0) + 1

        by_match = {}
        for o in outcomes:
            mt = o.get("match_type", "Unknown")
            by_match[mt] = by_match.get(mt, 0) + 1

        confidences = [o.get("confidence") or 0 for o in outcomes]
        avg_conf = round(sum(confidences) / total, 1) if confidences else 0

        return {
            "total": total,
            "by_decision": by_decision,
            "by_match_type": by_match,
            "avg_confidence": avg_conf,
            "auto_created": by_decision.get(WorkOrderDecision.AUTO_CREATE, 0),
            "pending_review": by_decision.get(WorkOrderDecision.REVIEW, 0),
            "no_action": by_decision.get(WorkOrderDecision.NO_ACTION, 0),
        }

    def approve_review(self, item_id: str, reviewer: str = "operator", notes: str = "") -> dict:
        """
        Approve a pending review item:
        1. Update review queue status
        2. Re-trigger IFM Hub API call
        3. Log to training data (positive label)
        Returns the created work order payload.
        """
        self.queue.approve(item_id, reviewer=reviewer, notes=notes)
        item = self.queue.get_by_id(item_id)
        if item and item.get("payload_json"):
            payload = json.loads(item["payload_json"])
            response = self.creator._call_api(payload)
            self._append_training_record(item, label="approved")
            return {"status": "created", "payload": payload, "response": response}
        return {"status": "error", "message": "Item not found or payload missing"}

    def reject_review(self, item_id: str, reviewer: str = "operator", notes: str = "") -> dict:
        self.queue.reject(item_id, reviewer=reviewer, notes=notes)
        item = self.queue.get_by_id(item_id)
        if item:
            self._append_training_record(item, label="rejected")
        return {"status": "rejected", "item_id": item_id}

    # ── Private ────────────────────────────────────────────────────────────────

    def _handle_event(self, event: SensorEvent) -> dict:
        """Called by ingestor for every event. Classify → decide → act."""
        classification = self.classifier.classify(event)
        outcome = self.creator.process_event(event, classification)

        decision = outcome["decision"]
        if decision == WorkOrderDecision.REVIEW:
            review_id = self.queue.enqueue(outcome)
            outcome["review_id"] = review_id

        self._outcomes.append(outcome)
        self._log_outcome(outcome)

        if self._on_outcome:
            try:
                self._on_outcome(outcome)
            except Exception as exc:
                logger.warning("Outcome callback failed: %s", exc)

        return outcome

    def _log_outcome(self, outcome: dict):
        """Append outcome to JSONL log file for audit trail."""
        try:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(outcome) + "\n")
        except Exception as exc:
            logger.debug("Could not write outcome log: %s", exc)

    def _append_training_record(self, review_item: dict, label: str):
        """
        Write approved/rejected review to training dataset
        for active learning (future model improvement).
        """
        try:
            train_path = Path(__file__).parent.parent / "data" / "training" / "active_learning.jsonl"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "asset_id":      review_item.get("asset_id"),
                "asset_name":    review_item.get("asset_name"),
                "alert_type":    review_item.get("alert_type"),
                "match_type":    review_item.get("match_type"),
                "confidence":    review_item.get("confidence"),
                "label":         label,   # "approved" or "rejected"
                "reviewed_at":   review_item.get("reviewed_at"),
                "reviewer":      review_item.get("reviewer"),
                "notes":         review_item.get("review_notes"),
            }
            with open(train_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("Could not write training record: %s", exc)


# ── Module-level singleton ────────────────────────────────────────────────────

_pipeline: Optional[SensorToWorkOrderPipeline] = None


def get_pipeline(use_llm: bool = True, demo_mode: bool = True) -> SensorToWorkOrderPipeline:
    """Return module-level singleton pipeline (lazy init)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SensorToWorkOrderPipeline(use_llm=use_llm, demo_mode=demo_mode)
    return _pipeline


def reset_pipeline():
    global _pipeline
    _pipeline = None
