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
        """Process multiple sensor events with a small inter-event delay to avoid rate limiting."""
        import time
        results = []
        for i, e in enumerate(raw_events):
            results.append(self.process_raw_event(e))
            if i < len(raw_events) - 1:
                time.sleep(0.3)   # 300ms between events → ~3 req/s, well within Azure limits
        return results

    def process_correlated_batch(self, raw_events: List[dict]) -> List[dict]:
        """
        Process alerts with LLM-based correlation.

        Uses GPT-5.5 to identify groups of related alerts (e.g. multiple devices
        offline in the same location → power outage) and creates ONE work order
        per correlated group instead of one per alert.

        Returns list of outcome dicts (one per group + one per ungrouped alert).
        """
        import time

        if len(raw_events) <= 1:
            return self.process_batch(raw_events)

        # Step 1: Ask the LLM to group correlated alerts
        groups = _correlate_alerts_with_llm(raw_events)

        results = []
        for group in groups:
            alerts_in_group = group["alerts"]
            if len(alerts_in_group) == 1:
                # Single alert → normal processing
                results.append(self.process_raw_event(alerts_in_group[0]))
            else:
                # Correlated group → process as one combined event
                outcome = self._process_correlated_group(
                    alerts_in_group,
                    group.get("root_cause", ""),
                    group.get("recommended_service", ""),
                )
                results.append(outcome)
            if len(groups) > 1:
                time.sleep(0.3)
        return results

    def _process_correlated_group(
        self,
        raw_alerts: List[dict],
        root_cause: str,
        recommended_service: str,
    ) -> dict:
        """
        Process a group of correlated alerts as a single work order.
        The group is treated as one combined event with the highest severity.
        """
        from pipeline.work_order_creator import build_ifm_request_payload, WorkOrderDecision

        # Ingest all events to validate them
        events = [self.ingestor.ingest(r) for r in raw_alerts]

        # Pick highest severity as the group severity
        _SEV_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        events_sorted = sorted(
            events, key=lambda e: _SEV_ORDER.get(e.severity, 0), reverse=True
        )
        primary_event = events_sorted[0]

        # Classify using the recommended service hint from LLM
        classification = self.classifier.classify(primary_event)

        # Override with LLM correlation insight if it gives higher confidence
        if recommended_service:
            from pipeline.service_classifier import _load_catalog
            catalog = _load_catalog()
            for sc in catalog:
                if (recommended_service.lower() in sc["name"].lower()
                        or sc["id"] == recommended_service):
                    classification = {
                        "service_classification_id": sc["id"],
                        "service_classification_name": sc["name"],
                        "match_type": "Correlated Group (LLM)",
                        "confidence": max(classification.get("confidence", 0), 88.0),
                        "reasoning": root_cause,
                    }
                    break

        # Build combined description
        asset_names = [r.get("asset_name", "?") for r in raw_alerts]
        alert_types = list({r.get("alert_type", "?") for r in raw_alerts})
        combined_description = (
            f"[Correlated Group — {len(raw_alerts)} alerts] "
            f"Root cause: {root_cause}. "
            f"Assets: {', '.join(asset_names)}. "
            f"Alert types: {', '.join(alert_types)}."
        )

        # Force the classification match_type to reflect correlation
        classification["match_type"] = classification.get("match_type", "Correlated Group (LLM)")
        if "Correlated" not in classification["match_type"]:
            classification["match_type"] = f"{classification['match_type']} [Correlated]"
        classification["reasoning"] = root_cause or classification.get("reasoning", "")

        # Build outcome through the normal creator (gets decision, payload, etc.)
        outcome = self.creator.process_event(primary_event, classification)

        # Enrich outcome with correlation metadata
        outcome["correlated"] = True
        outcome["correlated_count"] = len(raw_alerts)
        outcome["correlated_assets"] = asset_names
        outcome["correlated_alert_types"] = alert_types
        outcome["root_cause"] = root_cause
        outcome["description_override"] = combined_description

        # Override description in payload if it was created
        if outcome.get("request_payload"):
            outcome["request_payload"]["description"] = combined_description

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


# ── LLM Alert Correlation ─────────────────────────────────────────────────────

_CORRELATION_SYSTEM_PROMPT = """\
You are an IFM (Integrated Facilities Management) sensor alert correlation engine.

Given a batch of sensor alerts, identify which alerts are likely caused by the same root cause
and should be grouped into a single work order request.

Rules:
- Multiple devices offline/faulting in the SAME location → likely power outage or network failure
- Multiple HVAC devices in the same zone with related faults → likely zone system failure
- Alerts from unrelated assets/locations/categories should NOT be grouped
- Each alert must appear in exactly one group
- Single unrelated alerts form a group of 1

Output ONLY valid JSON with this structure:
{
  "groups": [
    {
      "alert_indices": [0, 2, 5],
      "root_cause": "Power outage in Building A Floor B1 — all devices lost power simultaneously",
      "recommended_service": "Electrical - Power Failure / Outage"
    },
    {
      "alert_indices": [1],
      "root_cause": "",
      "recommended_service": ""
    }
  ]
}
"""


def _correlate_alerts_with_llm(raw_events: List[dict]) -> List[dict]:
    """
    Use GPT-5.5 to identify correlated alert groups.
    Returns a list of group dicts, each with 'alerts', 'root_cause', 'recommended_service'.
    Falls back to no-grouping (each alert alone) if LLM is unavailable.
    """
    from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer

    # Build concise summary for LLM
    alert_summaries = []
    for i, ev in enumerate(raw_events):
        alert_summaries.append(
            f"[{i}] asset_name={ev.get('asset_name','?')} "
            f"asset_type={ev.get('asset_type','?')} "
            f"alert_type={ev.get('alert_type','?')} "
            f"severity={ev.get('severity','?')} "
            f"building={ev.get('building','?')} "
            f"floor={ev.get('floor','?')} "
            f"room={ev.get('room','?')}"
        )
    user_msg = "Alerts:\n" + "\n".join(alert_summaries) + "\n\nGroup correlated alerts. JSON only."

    # Try LLM
    client = _get_correlation_llm_client()
    if not client:
        logger.warning("LLM client unavailable for correlation — processing alerts individually.")
        return [{"alerts": [ev], "root_cause": "", "recommended_service": ""} for ev in raw_events]

    _model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
    try:
        with _MtTimer() as _t:
            response = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": _CORRELATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=1024,
            )
        raw_resp = response.choices[0].message.content
        parsed = json.loads(raw_resp)
        groups_raw = parsed.get("groups", [])

        _mt_record(
            model=_model, purpose="alert_correlation",
            pipeline="sensor_pipeline", node="correlate_alerts",
            success=True, latency_ms=_t.elapsed_ms,
            tokens_in_est=len(user_msg) // 4,
            tokens_out_est=len(raw_resp) // 4,
            extra={"num_alerts": len(raw_events), "num_groups": len(groups_raw)},
        )
        logger.info(
            "[LLM] Alert correlation: %d alerts → %d groups in %dms",
            len(raw_events), len(groups_raw), _t.elapsed_ms,
        )

        # Map indices back to raw events
        result = []
        used_indices = set()
        for g in groups_raw:
            indices = g.get("alert_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(raw_events)]
            if valid_indices:
                result.append({
                    "alerts": [raw_events[i] for i in valid_indices],
                    "root_cause": g.get("root_cause", ""),
                    "recommended_service": g.get("recommended_service", ""),
                })
                used_indices.update(valid_indices)

        # Any alerts not included by LLM → add as individual groups
        for i, ev in enumerate(raw_events):
            if i not in used_indices:
                result.append({"alerts": [ev], "root_cause": "", "recommended_service": ""})

        return result

    except Exception as exc:
        logger.error("LLM correlation failed: %s — falling back to individual processing", exc)
        _mt_record(model=_model, purpose="alert_correlation",
                   pipeline="sensor_pipeline", node="correlate_alerts", success=False)
        return [{"alerts": [ev], "root_cause": "", "recommended_service": ""} for ev in raw_events]


def _get_correlation_llm_client():
    """Return Azure OpenAI client for correlation, or None if not configured."""
    endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key     = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        return None
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    except Exception as exc:
        logger.warning("Could not init AzureOpenAI client for correlation: %s", exc)
        return None


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
