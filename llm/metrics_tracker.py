"""
llm/metrics_tracker.py
=======================
Singleton LLM Usage Metrics Tracker — SFM ↔ IFM AI Platform

Instruments every LLM call across all pipeline stages and records:
  • timestamp, model, model_display
  • purpose  — what the LLM was asked to do
  • pipeline — which pipeline called it ("asset_mapping" | "sensor_pipeline" | "chat")
  • node     — which graph node / function triggered it
  • success, latency_ms
  • tokens_in_est, tokens_out_est  (character-based rough estimate if not returned by SDK)

All records are appended to data/llm_metrics.jsonl (append-only).
The in-memory list powers the real-time dashboard without a file read.

Models tracked:
  • Azure OpenAI GPT-5.5  (deployment: gpt-5.5_1)
  • Azure Anthropic Claude Sonnet 4.6 (deployment: claude-sonnet-4-6)

Usage:
    from llm.metrics_tracker import record, get_summary, Timer

    with Timer() as t:
        response = llm_client.chat(...)
    record(
        model="gpt-5.5_1",
        purpose="asset_matching_reason",
        pipeline="asset_mapping",
        node="llm_reason",
        success=True,
        latency_ms=t.elapsed_ms,
        tokens_in_est=len(prompt) // 4,
        tokens_out_est=len(response) // 4,
    )
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Storage ───────────────────────────────────────────────────────────────────
_METRICS_PATH = Path(__file__).parent.parent / "data" / "llm_metrics.jsonl"
_lock = Lock()
_in_memory: List[Dict[str, Any]] = []

# ── Human-readable model display names ───────────────────────────────────────
MODEL_DISPLAY: Dict[str, str] = {
    "gpt-5.5_1":          "Azure OpenAI GPT-5.5",
    "gpt-5.5":            "Azure OpenAI GPT-5.5",
    "gpt-4o":             "Azure OpenAI GPT-4o",
    "claude-sonnet-4-6":  "Azure Anthropic Claude Sonnet 4.6",
    "claude-3-5-sonnet":  "Azure Anthropic Claude 3.5 Sonnet",
}

# ── Purpose labels (for dashboard legend) ────────────────────────────────────
PURPOSE_LABELS: Dict[str, str] = {
    "asset_matching_reason":   "Asset Matching — LLM Reason Node",
    "asset_matching_partial":  "Asset Matching — Partial LLM Verify",
    "service_classification":  "Service Classification — Tier 3 LLM",
    "service_partial_verify":  "Service Classification — Tier 2 LLM Verify",
    "chat_agent":              "AI Chat Agent (LangChain ReAct)",
    "chat_direct":             "AI Chat Agent (Direct LLM)",
}


# ── Core record function ──────────────────────────────────────────────────────

def record(
    model: str,
    purpose: str,
    pipeline: str,
    node: str = "",
    success: bool = True,
    latency_ms: int = 0,
    tokens_in_est: int = 0,
    tokens_out_est: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append one LLM call record. Thread-safe.
    Called immediately after every LLM API call completes.
    """
    entry: Dict[str, Any] = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "model":           model,
        "model_display":   MODEL_DISPLAY.get(model, model),
        "purpose":         purpose,
        "purpose_label":   PURPOSE_LABELS.get(purpose, purpose),
        "pipeline":        pipeline,
        "node":            node,
        "success":         success,
        "latency_ms":      latency_ms,
        "tokens_in_est":   tokens_in_est,
        "tokens_out_est":  tokens_out_est,
    }
    if extra:
        entry.update(extra)

    with _lock:
        _in_memory.append(entry)
        try:
            _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_METRICS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("LLM metrics write failed: %s", exc)


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_all() -> List[Dict[str, Any]]:
    """Return all in-memory records, newest first."""
    with _lock:
        return list(reversed(_in_memory))


# Alias used by Tab 6
def get_records() -> List[Dict[str, Any]]:
    """Return all in-memory records (oldest first)."""
    with _lock:
        return list(_in_memory)


def get_summary() -> Dict[str, Any]:
    """
    Aggregate statistics for dashboard KPIs and charts.
    Returns dict with:
      total_calls, success_rate, avg_latency_ms, total_tokens_est,
      by_model  — {"deployment_name": {"calls": n, "success": n, "total_latency_ms": n}}
      by_purpose — {"purpose_code": {"calls": n}}
      by_pipeline — {"pipeline_name": {"calls": n}}
      timeline  — [{"timestamp": str, "model": str, "latency_ms": int, "success": bool}, ...]
    """
    with _lock:
        records = list(_in_memory)

    if not records:
        return {
            "total_calls":      0,
            "success_rate":     100.0,
            "avg_latency_ms":   0,
            "total_tokens_est": 0,
            "by_model":         {},
            "by_purpose":       {},
            "by_pipeline":      {},
            "timeline":         [],
        }

    total     = len(records)
    successes = sum(1 for r in records if r["success"])
    avg_lat   = round(sum(r["latency_ms"] for r in records) / total)
    total_tok = sum(r.get("tokens_in_est", 0) + r.get("tokens_out_est", 0) for r in records)

    # by_model keyed on raw deployment name — Tab 6 maps display via MODEL_DISPLAY
    by_model: Dict[str, Dict[str, Any]] = {}
    by_purpose: Dict[str, Dict[str, Any]] = {}
    by_pipeline: Dict[str, Dict[str, Any]] = {}

    for r in records:
        m   = r["model"]
        p   = r.get("purpose", "unknown")
        pl  = r.get("pipeline", "unknown")
        suc = bool(r.get("success", False))
        lat = r.get("latency_ms", 0)

        if m not in by_model:
            by_model[m] = {"calls": 0, "success": 0, "total_latency_ms": 0}
        by_model[m]["calls"] += 1
        by_model[m]["success"] += int(suc)
        by_model[m]["total_latency_ms"] += lat

        if p not in by_purpose:
            by_purpose[p] = {"calls": 0}
        by_purpose[p]["calls"] += 1

        if pl not in by_pipeline:
            by_pipeline[pl] = {"calls": 0}
        by_pipeline[pl]["calls"] += 1

    timeline = [
        {
            "timestamp":  r["timestamp"],
            "model":      r["model"],
            "purpose":    r.get("purpose", ""),
            "latency_ms": r["latency_ms"],
            "success":    r["success"],
        }
        for r in records[-100:]   # last 100 for chart
    ]

    return {
        "total_calls":      total,
        "success_rate":     round(successes / total * 100, 1),
        "avg_latency_ms":   avg_lat,
        "total_tokens_est": total_tok,
        "by_model":         by_model,
        "by_purpose":       by_purpose,
        "by_pipeline":      by_pipeline,
        "timeline":         timeline,
    }


def reset() -> None:
    """Clear in-memory metrics (does not delete the JSONL file)."""
    with _lock:
        _in_memory.clear()


# ── Timer context manager ─────────────────────────────────────────────────────

class Timer:
    """
    Context manager for wall-clock latency measurement.

    Usage:
        with Timer() as t:
            call_llm(...)
        print(t.elapsed_ms)   # int milliseconds
    """
    def __init__(self) -> None:
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = int((time.monotonic() - self._t0) * 1000)
