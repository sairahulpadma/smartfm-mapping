"""
chat_agent.py
LangChain-based chat agent that answers questions about mapping results.
Uses pandas DataFrame tools so it can query actual data in real time.
"""

from __future__ import annotations

import os
import json
import pandas as pd
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic

from llm.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


# ── Shared state: results DataFrame injected at runtime ──────────────────────
_results_df: Optional[pd.DataFrame] = None


def set_results(df: pd.DataFrame):
    """Call this once after mapping is complete to give the agent data access."""
    global _results_df
    _results_df = df.copy()


# ── Tools the agent can call ──────────────────────────────────────────────────

@tool
def get_summary_stats(query: str = "") -> str:
    """
    Returns high-level statistics about the mapping results:
    total records, match type distribution, average confidence, top unmatched assets.
    The query argument is ignored – always returns full stats.
    """
    if _results_df is None:
        return "No mapping results loaded yet."

    df = _results_df
    total = len(df)
    counts = df["match_type"].value_counts().to_dict()
    avg_conf = round(df["confidence"].mean(), 1)
    unmatched = df[df["match_type"] == "No Match"][["sfm_nav_name", "confidence"]]

    return json.dumps({
        "total_assets": total,
        "match_distribution": counts,
        "average_confidence": avg_conf,
        "unmatched_count": int(counts.get("No Match", 0)),
        "unmatched_assets": unmatched["sfm_nav_name"].tolist()[:20],
    })


@tool
def query_assets(filter_expr: str) -> str:
    """
    Filter the mapping results DataFrame using a pandas query string.
    Examples:
      - "match_type == 'No Match'"
      - "confidence < 50"
      - "matched_building == 'Pleasanton Campus - Building A'"
      - "match_type.str.contains('Partial')"
    Returns top 20 matching rows as JSON.
    """
    if _results_df is None:
        return "No mapping results loaded yet."
    try:
        result = _results_df.query(filter_expr, engine="python")
        cols = ["sfm_nav_name", "matched_asset_name", "match_type",
                "confidence", "matched_building"]
        return result[cols].head(20).to_json(orient="records", indent=2)
    except Exception as e:
        return f"Query error: {e}. Try a simpler filter expression."


@tool
def get_building_analysis(building_name: str = "") -> str:
    """
    Returns match quality breakdown for a specific building, or all buildings
    if building_name is empty/all.
    """
    if _results_df is None:
        return "No mapping results loaded yet."

    df = _results_df.copy()
    df["building_clean"] = df["matched_building"].fillna("Unknown")

    if building_name and building_name.lower() != "all":
        df = df[df["building_clean"].str.contains(building_name, case=False, na=False)]

    grouped = df.groupby("building_clean").agg(
        total=("sfm_nav_name", "count"),
        perfect=("match_type", lambda x: (x.str.contains("Perfect", na=False)).sum()),
        partial=("match_type", lambda x: (x.str.contains("Partial", na=False)).sum()),
        no_match=("match_type", lambda x: (x == "No Match").sum()),
        avg_confidence=("confidence", "mean"),
    ).round(1).reset_index()

    return grouped.to_json(orient="records", indent=2)


@tool
def get_low_confidence_assets(threshold: str = "50") -> str:
    """
    Returns all assets with confidence below the given threshold (default 50).
    Good for finding 'bad' or problematic mappings.
    """
    if _results_df is None:
        return "No mapping results loaded yet."
    try:
        t = float(threshold)
    except ValueError:
        t = 50.0
    result = _results_df[_results_df["confidence"] < t]
    cols = ["sfm_nav_name", "matched_asset_name", "match_type",
            "confidence", "matched_building"]
    return result[cols].head(30).to_json(orient="records", indent=2)


# ── Build the LangChain agent ─────────────────────────────────────────────────

_TOOLS = [get_summary_stats, query_assets, get_building_analysis, get_low_confidence_assets]

_TOOL_NAMES = ", ".join([t.name for t in _TOOLS])
_TOOL_DESCS = "\n".join([f"{t.name}: {t.description}" for t in _TOOLS])

_REACT_TEMPLATE = f"""{SYSTEM_PROMPT}

You have access to these tools:
{_TOOL_DESCS}

Use this format:
Question: the user's question
Thought: what to do
Action: tool name (one of: {_TOOL_NAMES})
Action Input: input to the tool
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: your answer to the user

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}"""

_PROMPT = PromptTemplate.from_template(_REACT_TEMPLATE)


def _build_llm():
    """Build LLM – tries Anthropic first, falls back to Azure OpenAI."""
    anthropic_key = os.getenv("AZURE_OPENAI_API_KEY", "")  # reuse same key for demo
    try:
        import httpx
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            base_url=os.getenv("ANTHROPIC_ENDPOINT",
                               "https://admv-mogidbp0-eastus2.services.ai.azure.com/anthropic/"),
            http_client=httpx.Client(verify=False),
            max_tokens=2048,
        )
    except Exception:
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            max_tokens=2048,
        )


def get_chat_response(user_message: str, chat_history: list[dict]) -> str:
    """
    Get a chat response for the user's question about the mapping results.
    Falls back to direct data query if LLM is unavailable.
    """
    if _results_df is None:
        return "Please run the asset mapping first before asking questions about the data."

    try:
        llm = _build_llm()
        agent = create_react_agent(llm, _TOOLS, _PROMPT)
        executor = AgentExecutor(
            agent=agent, tools=_TOOLS,
            verbose=False, handle_parsing_errors=True,
            max_iterations=5,
        )
        result = executor.invoke({"input": user_message})
        return result.get("output", "I could not generate a response.")
    except Exception as e:
        # Graceful fallback: answer directly from data
        return _fallback_answer(user_message)


def _fallback_answer(question: str) -> str:
    """Simple keyword-based fallback when LLM is not available."""
    if _results_df is None:
        return "No data available."
    df = _results_df
    q = question.lower()

    if "no match" in q or "unmatched" in q:
        unmatched = df[df["match_type"] == "No Match"]["sfm_nav_name"].tolist()
        return f"**Unmatched assets ({len(unmatched)}):**\n" + "\n".join(f"- {a}" for a in unmatched)

    if "summary" in q or "overview" in q or "stats" in q:
        counts = df["match_type"].value_counts()
        lines = [f"**Total assets:** {len(df)}"]
        for k, v in counts.items():
            pct = round(v / len(df) * 100, 1)
            lines.append(f"- {k}: {v} ({pct}%)")
        lines.append(f"**Average confidence:** {round(df['confidence'].mean(), 1)}%")
        return "\n".join(lines)

    if "low confidence" in q or "bad" in q or "problem" in q:
        low = df[df["confidence"] < 50][["sfm_nav_name", "confidence", "match_type"]]
        return f"**Low confidence assets (< 50%):**\n{low.to_string(index=False)}"

    return (
        "I can answer questions like:\n"
        "- Which assets have no match?\n"
        "- Show me low confidence matches\n"
        "- Give me a summary of mapping quality\n"
        "- Which building has the most unmatched assets?"
    )
