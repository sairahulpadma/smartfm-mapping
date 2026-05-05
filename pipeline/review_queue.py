"""
review_queue.py
Human review queue for Partial Match and LLM Reasoned work order requests.

Storage: SQLite (production swap: Azure Service Bus + CosmosDB).
Schema:
  review_items(id, event_id, asset_id, classification_json, payload_json,
               match_type, confidence, status, reviewer, reviewed_at, created_at)

Status lifecycle:
  PENDING → APPROVED  → work order created via IFM Hub API
         → REJECTED   → logged, no request created
         → ESCALATED  → sent to senior reviewer
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "review_queue.db"


# ── Status constants ───────────────────────────────────────────────────────────

class ReviewStatus:
    PENDING   = "PENDING"
    APPROVED  = "APPROVED"
    REJECTED  = "REJECTED"
    ESCALATED = "ESCALATED"


# ── DB helpers ────────────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create table if it doesn't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_items (
                id                  TEXT PRIMARY KEY,
                event_id            TEXT,
                asset_id            TEXT,
                asset_name          TEXT,
                alert_type          TEXT,
                severity            TEXT,
                building            TEXT,
                classification_json TEXT,
                payload_json        TEXT,
                match_type          TEXT,
                confidence          REAL,
                status              TEXT DEFAULT 'PENDING',
                reviewer            TEXT,
                reviewed_at         TEXT,
                review_notes        TEXT,
                created_at          TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON review_items(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asset  ON review_items(asset_id)")


# ── Queue class ────────────────────────────────────────────────────────────────

class ReviewQueue:
    """
    Manages the human review queue for ambiguous work order requests.

    Usage:
        queue = ReviewQueue()
        item_id = queue.enqueue(outcome)
        items = queue.get_pending()
        queue.approve(item_id, reviewer="John", notes="Confirmed HVAC issue")
    """

    def __init__(self):
        init_db()

    def enqueue(self, outcome: dict) -> str:
        """
        Add a new item to the review queue.
        outcome is the dict returned by WorkOrderCreator._outcome().
        Returns the review item ID.
        """
        item_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute("""
                INSERT INTO review_items (
                    id, event_id, asset_id, asset_name, alert_type, severity,
                    building, classification_json, payload_json, match_type,
                    confidence, status, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                item_id,
                outcome.get("event_id", ""),
                outcome.get("asset_id", ""),
                outcome.get("asset_name", ""),
                outcome.get("alert_type", ""),
                outcome.get("severity", ""),
                outcome.get("building", ""),
                json.dumps(outcome.get("request_payload") or {}),
                json.dumps(outcome.get("request_payload") or {}),
                outcome.get("match_type", ""),
                float(outcome.get("confidence") or 0),
                ReviewStatus.PENDING,
                now,
            ))
        logger.info("Enqueued review item %s for asset %s", item_id, outcome.get("asset_id"))
        return item_id

    def get_pending(self, limit: int = 100) -> List[dict]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM review_items WHERE status=? ORDER BY confidence ASC, created_at ASC LIMIT ?",
                (ReviewStatus.PENDING, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self, status: Optional[str] = None, limit: int = 200) -> List[dict]:
        with _db() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM review_items WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM review_items ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, item_id: str) -> Optional[dict]:
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM review_items WHERE id=?", (item_id,)
            ).fetchone()
        return dict(row) if row else None

    def approve(self, item_id: str, reviewer: str = "system", notes: str = "") -> bool:
        """
        Approve an item → triggers work order creation in IFM Hub.
        Returns True if status updated.
        """
        return self._update_status(item_id, ReviewStatus.APPROVED, reviewer, notes)

    def reject(self, item_id: str, reviewer: str = "system", notes: str = "") -> bool:
        return self._update_status(item_id, ReviewStatus.REJECTED, reviewer, notes)

    def escalate(self, item_id: str, reviewer: str = "system", notes: str = "") -> bool:
        return self._update_status(item_id, ReviewStatus.ESCALATED, reviewer, notes)

    def get_stats(self) -> dict:
        with _db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt, AVG(confidence) as avg_conf "
                "FROM review_items GROUP BY status"
            ).fetchall()
        stats: Dict[str, Any] = {
            "total": 0,
            "by_status": {},
        }
        for r in rows:
            stats["by_status"][r["status"]] = {
                "count": r["cnt"],
                "avg_confidence": round(r["avg_conf"] or 0, 1),
            }
            stats["total"] += r["cnt"]
        return stats

    def seed_demo_data(self):
        """Insert demo review items for UI showcase."""
        demo_outcomes = [
            {
                "event_id": str(uuid.uuid4()), "asset_id": "SFM-AHU-B1-01",
                "asset_name": "AHU-1 Level B1", "alert_type": "temperature_high",
                "severity": "HIGH", "building": "HQ Building A",
                "match_type": "Partial Match", "confidence": 67.3,
                "request_payload": {
                    "id": str(uuid.uuid4()),
                    "description": "Partial match — AHU temperature_high → HVAC Cooling",
                    "serviceClassificationId": "4aa86b28-c506-11ed-afa1-0242ac120002",
                },
            },
            {
                "event_id": str(uuid.uuid4()), "asset_id": "SFM-THERM-FL2-01",
                "asset_name": "Thermostat Zone 2A", "alert_type": "setpoint_deviation",
                "severity": "MEDIUM", "building": "HQ Building A",
                "match_type": "LLM Reasoned", "confidence": 76.0,
                "request_payload": {
                    "id": str(uuid.uuid4()),
                    "description": "LLM Reasoned — thermostat setpoint deviation → Controls Issue",
                    "serviceClassificationId": "6cc08d4a-e728-12ee-c0c3-2464ce342224",
                },
            },
            {
                "event_id": str(uuid.uuid4()), "asset_id": "SFM-PUMP-CWR-01",
                "asset_name": "Chilled Water Pump 1", "alert_type": "vibration_high",
                "severity": "MEDIUM", "building": "HQ Building A",
                "match_type": "Partial Match", "confidence": 58.1,
                "request_payload": {
                    "id": str(uuid.uuid4()),
                    "description": "Partial match — pump vibration → Mechanical Vibration",
                    "serviceClassificationId": "i882k9g6-0j40-140k-82o5-4686o056464k",
                },
            },
        ]
        existing = {r["asset_id"] for r in self.get_all()}
        for o in demo_outcomes:
            if o["asset_id"] not in existing:
                self.enqueue(o)

    def _update_status(
        self, item_id: str, status: str, reviewer: str, notes: str
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            cur = conn.execute(
                "UPDATE review_items SET status=?, reviewer=?, reviewed_at=?, review_notes=? WHERE id=?",
                (status, reviewer, now, notes, item_id),
            )
        return cur.rowcount > 0
