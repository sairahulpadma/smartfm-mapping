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

_SYSTEM_PROMPT = """You are an expert IFM (Integrated Facility Management) service classification specialist.

Your task: given a building asset's type and its sensor alert, determine the BEST service classification
from the provided candidates to route a work order to the correct IFM team.

Rules:
1. Pick ONLY from the provided candidate list (use the "id" field exactly).
2. Assign a confidence score (0-100) reflecting how certain you are.
3. Confidence 85+ means Perfect, 50-84 means Partial. Never return below 30.
4. Provide a brief, precise reasoning (1-2 sentences).
5. Return ONLY a valid JSON object with keys: chosen_id, confidence, reasoning.

Example output:
{"chosen_id": "4aa86b28-c506-11ed-afa1-0242ac120002", "confidence": 78, "reasoning": "AHU with temperature_high maps to HVAC cooling system failure. Partial confidence as no make/model context provided."}
"""

# ── Catalog reference (loaded once) ──────────────────────────────────────────

from pathlib import Path
import json as _json

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

    user_msg = f"""Asset Type: {asset_type}
Alert Type: {alert_type.replace('_', ' ').title()}
Description: {description}

Top fuzzy-matched service classification candidates:
{candidate_text}

Which service classification ID best fits this alert? Return JSON only."""

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        chosen_id  = parsed.get("chosen_id", "")
        confidence = float(parsed.get("confidence", 60))
        reasoning  = parsed.get("reasoning", "LLM classification.")

        catalog = _get_catalog()
        sc = catalog.get(chosen_id)
        if sc:
            return sc, confidence, reasoning

        logger.warning("LLM returned unknown id: %s", chosen_id)
        return None, 0.0, ""

    except Exception as exc:
        logger.error("LLM classify_with_llm failed: %s", exc)
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
            temperature=0,
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
