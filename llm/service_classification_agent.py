"""
service_classification_agent.py
LLM-based service classification using Azure OpenAI GPT-5.5.

Called by ServiceClassifier when fuzzy matching yields ambiguous results
(Tier 3 fallback). The agent:
  1. Receives asset_type + alert_type + top fuzzy candidates
  2. Returns the best service classification with reasoning
  3. Returns (sc_dict, confidence, reasoning) tuple
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ── LLM System prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "IFM service classification specialist. "
    "Pick the best service classification from the candidates. "
    "Rules: use exact candidate id; confidence 85+=Perfect, 50-84=Partial, never <30; "
    "1-2 sentence reasoning. "
    'Output ONLY JSON: {"chosen_id":"...","confidence":0-100,"reasoning":"..."}'
)

# ── Catalog reference (loaded once) ──────────────────────────────────────────

from pathlib import Path
import json as _json
from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer

_CATALOG: dict = {}
_CATALOG_PATH = Path(__file__).parent.parent / "data" / "service_catalog.json"

def _get_catalog() -> dict:
    global _CATALOG
    if not _CATALOG:
        try:
            with open(_CATALOG_PATH) as f:
                raw = _json.load(f)
            _CATALOG = {sc["id"]: sc for sc in raw["service_classifications"]}
        except Exception:
            pass
    return _CATALOG


# ── Main function ─────────────────────────────────────────────────────────────

def classify_with_llm(
    asset_type: str,
    alert_type: str,
    description: str,
    candidates: List[dict],
) -> Tuple[Optional[dict], float, str]:
    """
    Ask the LLM to pick the best service classification.

    Parameters
    ----------
    asset_type  : e.g. "AHU"
    alert_type  : e.g. "temperature_high"
    description : asset name / additional context
    candidates  : list of {id, name, score} dicts (top fuzzy matches)

    Returns
    -------
    (service_classification_dict, confidence, reasoning)
    Returns (None, 0, "") on failure.
    """
    client = _get_openai_client()
    if not client:
        logger.warning("LLM client unavailable — service classification LLM fallback skipped.")
        return None, 0.0, ""

    candidate_text = "\n".join(
        f"  - id={c['id']}, name={c['name']}, fuzzy_score={c['score']:.1f}"
        for c in candidates
    )

    # Truncate description to 80 chars to avoid padding tokens
    desc_short = (description or "")[:80]
    user_msg = (
        f"Asset:{asset_type} Alert:{alert_type} Desc:{desc_short}\n"
        f"Candidates:\n{candidate_text}\n"
        f"Best match? JSON only."
    )

    _model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
    try:
        with _MtTimer() as _t:
            response = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=256,
            )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        chosen_id  = parsed.get("chosen_id", "")
        confidence = float(parsed.get("confidence", 60))
        reasoning  = parsed.get("reasoning", "LLM classification.")

        catalog = _get_catalog()
        sc = catalog.get(chosen_id)
        if sc:
            # ── Record successful LLM call ─────────────────────────────────
            _mt_record(
                model=_model, purpose="service_classification",
                pipeline="sensor_pipeline", node="tier3_llm_classify",
                success=True, latency_ms=_t.elapsed_ms,
                tokens_in_est=len(user_msg) // 4,
                tokens_out_est=len(raw) // 4,
                extra={"chosen_id": chosen_id, "confidence": confidence},
            )
            logger.info(
                "[LLM] Service classification → %s (%.1f%%) in %dms  model=%s",
                sc.get("name", chosen_id), confidence, _t.elapsed_ms, _model,
            )
            return sc, confidence, reasoning

        logger.warning("LLM returned unknown id: %s", chosen_id)
        _mt_record(model=_model, purpose="service_classification",
                   pipeline="sensor_pipeline", node="tier3_llm_classify",
                   success=False, latency_ms=_t.elapsed_ms)
        return None, 0.0, ""

    except Exception as exc:
        logger.error("LLM classify_with_llm failed: %s", exc)
        _mt_record(model=_model, purpose="service_classification",
                   pipeline="sensor_pipeline", node="tier3_llm_classify", success=False)
        return None, 0.0, ""


# ── LLM client factory ────────────────────────────────────────────────────────

def _get_openai_client():
    """Return Azure OpenAI client, or None if not configured."""
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key    = os.getenv("AZURE_OPENAI_API_KEY",  "")
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
        logger.warning("Could not init AzureOpenAI client: %s", exc)
        return None


# ── Batch classification via LangChain ReAct agent (optional) ─────────────────

def build_classification_agent(results_df=None):
    """
    Build a LangChain ReAct agent for interactive service classification Q&A.
    Returns agent_executor or None if LLM is unavailable.
    """
    try:
        from langchain_openai import AzureChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.tools import tool
        from langchain import hub
        import pandas as pd

        endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key     = os.getenv("AZURE_OPENAI_API_KEY",  "")
        deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not endpoint or not api_key:
            return None

        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=api_version,
        )

        @tool
        def get_service_catalog() -> str:
            """Returns the full service classification catalog as JSON."""
            try:
                with open(_CATALOG_PATH) as f:
                    return f.read()
            except Exception:
                return "{}"

        @tool
        def query_work_orders(query: str) -> str:
            """Run a pandas query on the work orders / sensor results DataFrame."""
            if results_df is None:
                return "No data available."
            try:
                df = results_df.query(query) if query else results_df
                return df.head(20).to_string()
            except Exception as exc:
                return f"Query error: {exc}"

        prompt = hub.pull("hwchase17/react")
        agent  = create_react_agent(llm, [get_service_catalog, query_work_orders], prompt)
        return AgentExecutor(agent=agent, tools=[get_service_catalog, query_work_orders],
                             verbose=False, handle_parsing_errors=True, max_iterations=5)
    except Exception as exc:
        logger.warning("Could not build classification agent: %s", exc)
        return None
