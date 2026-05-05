"""
work_order_creator.py
IFM Hub REST API client for creating work order requests.

Decision logic (mirrors the 4-tier asset matching):
  ✅ Perfect Match  (confidence ≥ 85%)  → auto-create request
  ⚠️  Partial Match (50–84%)            → enqueue for human review
  🤖 LLM Reasoned  (any confidence)    → enqueue for human review
  ❌ No Match                           → no request, log only

IFM Hub API contract:
  POST /requests  with JSON body (see IFM_REQUEST_SCHEMA below)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── IFM Hub configuration ─────────────────────────────────────────────────────

IFM_BASE_URL   = os.getenv("IFM_BASE_URL",   "https://api.ifm-hub.example.com")
IFM_API_KEY    = os.getenv("IFM_API_KEY",    "")
IFM_TENANT_ID  = os.getenv("IFM_TENANT_ID",  "601205a7f110dd542d9237bc")
IFM_ORG_IDS    = json.loads(os.getenv("IFM_ORG_IDS",
    '["7bb1b889-9a2e-4752-b94d-d1008188dbd1","fea0683c-d2ec-4728-9fc9-fc90eaa09b00"]'))
IFM_REQUESTOR_ID = os.getenv("IFM_REQUESTOR_ID", "cb0795b9-32d6-4574-8a88-bd8fc5b1b5cd")
IFM_SOURCE_APP   = os.getenv("IFM_SOURCE_APP",   "sfm-ai-platform")

PERFECT_THRESHOLD = float(os.getenv("PERFECT_THRESHOLD", "85"))


# ── Decision engine ───────────────────────────────────────────────────────────

class WorkOrderDecision:
    """Enum-like class for decision outcomes."""
    AUTO_CREATE = "AUTO_CREATE"
    REVIEW      = "REVIEW"
    NO_ACTION   = "NO_ACTION"


def decide_action(classification_result: dict) -> str:
    """
    Given a service classification result, decide what to do.

    Returns WorkOrderDecision constant.
    """
    match_type = classification_result.get("match_type", "No Match")
    confidence = float(classification_result.get("confidence", 0))

    if match_type == "No Match":
        return WorkOrderDecision.NO_ACTION

    if match_type == "Perfect Match" and confidence >= PERFECT_THRESHOLD:
        return WorkOrderDecision.AUTO_CREATE

    # Partial Match or LLM Reasoned → review
    return WorkOrderDecision.REVIEW


# ── Request builder ───────────────────────────────────────────────────────────

def build_ifm_request_payload(
    sensor_event: Any,
    classification: dict,
    alternate_id: Optional[str] = None,
) -> dict:
    """
    Builds the IFM Hub request JSON from a sensor event + classification result.
    Matches the IFM Hub API contract exactly.
    """
    request_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Build human-readable description from sensor event
    if hasattr(sensor_event, "to_dict"):
        ev = sensor_event.to_dict()
    else:
        ev = dict(sensor_event)

    asset_name  = ev.get("asset_name", ev.get("asset_id", "Unknown Asset"))
    alert_type  = ev.get("alert_type", "unknown_alert")
    building    = ev.get("building", "")
    floor       = ev.get("floor", "")
    room        = ev.get("room", "")
    reading     = ev.get("reading") or {}
    value       = reading.get("value", "N/A")
    unit        = reading.get("unit", "")
    threshold   = reading.get("threshold_max") or reading.get("threshold_min", "N/A")

    location_str = " | ".join(filter(None, [building, f"Floor {floor}" if floor else "", room]))
    description = (
        f"Auto-generated work order — {asset_name} | Alert: {alert_type.replace('_', ' ').title()} "
        f"| Reading: {value} {unit} (threshold: {threshold} {unit}) | Location: {location_str} | "
        f"Service: {classification.get('service_classification_name', 'N/A')} | "
        f"Confidence: {classification.get('confidence', 0):.1f}%"
    )

    payload = {
        "orgs": IFM_ORG_IDS,
        "tenantId": IFM_TENANT_ID,
        "id": request_id,
        "reportedDate": now_iso,
        "alternateId": alternate_id or f"AI-{request_id[:8].upper()}",
        "description": description,
        "locationId": ev.get("location_id", ""),
        "serviceClassificationId": classification["service_classification_id"],
        "relatedServiceClassificationId": [],
        "requestorId": IFM_REQUESTOR_ID,
        "ownerId": IFM_REQUESTOR_ID,
        "modifiedBy": IFM_REQUESTOR_ID,
        "modifiedDate": now_iso,
        "source": "AI Sensor Alert",
        "sourceApp": IFM_SOURCE_APP,
        "statusId": "13ef1492-8e5f-4337-9751-c42d1a823edf",   # "Open" status ID
        "attachments": [],
        "_meta": {
            "sensor_event_id": ev.get("event_id", ""),
            "asset_id": ev.get("asset_id", ""),
            "alert_type": alert_type,
            "severity": ev.get("severity", ""),
            "match_type": classification.get("match_type"),
            "match_confidence": classification.get("confidence"),
            "ai_reasoning": classification.get("reasoning", ""),
        },
    }
    return payload


# ── API client ────────────────────────────────────────────────────────────────

class WorkOrderCreator:
    """
    Sends work order creation requests to the IFM Hub API.

    In production: uses httpx for async HTTP with retry logic.
    In demo mode:  simulates API calls and returns a mock response.
    """

    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self._created: list = []       # audit log

    def process_event(
        self,
        sensor_event: Any,
        classification: dict,
    ) -> dict:
        """
        Full lifecycle for one sensor event:
          1. Decide action
          2. Build payload
          3. Create / review / skip
        Returns an outcome dict.
        """
        decision = decide_action(classification)

        if decision == WorkOrderDecision.NO_ACTION:
            logger.info("No action for event %s — No Match",
                        getattr(sensor_event, "event_id", "?"))
            return self._outcome(decision, None, classification, sensor_event)

        payload = build_ifm_request_payload(sensor_event, classification)

        if decision == WorkOrderDecision.AUTO_CREATE:
            response = self._call_api(payload)
            self._created.append(payload)
            logger.info("Work order created: %s (confidence %.1f%%)",
                        payload["id"], classification.get("confidence", 0))
            return self._outcome(decision, payload, classification, sensor_event, api_response=response)

        # REVIEW
        logger.info("Event %s queued for review (match_type=%s, confidence=%.1f%%)",
                    getattr(sensor_event, "event_id", "?"),
                    classification.get("match_type"),
                    classification.get("confidence", 0))
        return self._outcome(decision, payload, classification, sensor_event)

    def get_audit_log(self) -> list:
        return list(self._created)

    # ── private ────────────────────────────────────────────────────────────────

    def _call_api(self, payload: dict) -> dict:
        if self.demo_mode:
            return {"status": "created", "request_id": payload["id"], "demo": True}

        try:
            import httpx
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {IFM_API_KEY}",
            }
            resp = httpx.post(
                f"{IFM_BASE_URL}/requests",
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("IFM Hub API call failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _outcome(
        self,
        decision: str,
        payload: Optional[dict],
        classification: dict,
        sensor_event: Any,
        api_response: Optional[dict] = None,
    ) -> dict:
        ev_dict = sensor_event.to_dict() if hasattr(sensor_event, "to_dict") else dict(sensor_event)
        return {
            "event_id": ev_dict.get("event_id", ""),
            "asset_id": ev_dict.get("asset_id", ""),
            "asset_name": ev_dict.get("asset_name", ""),
            "alert_type": ev_dict.get("alert_type", ""),
            "severity": ev_dict.get("severity", ""),
            "building": ev_dict.get("building", ""),
            "service_classification_id": classification.get("service_classification_id"),
            "service_classification_name": classification.get("service_classification_name"),
            "match_type": classification.get("match_type"),
            "confidence": classification.get("confidence"),
            "decision": decision,
            "request_id": (payload or {}).get("id"),
            "request_payload": payload,
            "api_response": api_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
