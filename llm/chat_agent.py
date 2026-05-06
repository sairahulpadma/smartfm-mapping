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

try:
    from langchain.agents import AgentExecutor, create_react_agent
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False
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
    """
    Build LLM for the chat/ReAct agent.
    Routing strategy (token efficiency):
      • Claude Sonnet 4.6 via AnthropicFoundry  — nuanced Q&A  (primary)
      • GPT-5.5                                  — fallback if Claude creds missing/placeholder
    """
    anthropic_key      = os.getenv("AZURE_ANTHROPIC_API_KEY", "")
    anthropic_endpoint = os.getenv("AZURE_ANTHROPIC_ENDPOINT",
                                   "https://admv-mogidbp0-eastus2.services.ai.azure.com/anthropic/")
    # Only try Claude if the key is actually set (not the placeholder)
    if anthropic_key and not anthropic_key.startswith("<"):
        try:
            return ChatAnthropic(
                model=os.getenv("AZURE_ANTHROPIC_DEPLOYMENT", "claude-sonnet-4-6"),
                anthropic_api_key=anthropic_key,
                anthropic_api_url=anthropic_endpoint,
                max_tokens=1024,
            )
        except Exception:
            pass  # fall through to GPT
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        max_completion_tokens=1024,
    )


def get_chat_response(user_message: str, chat_history: list[dict]) -> str:
    """
    Get a chat response for the user's question about the mapping results.
    Falls back to direct data query if LLM is unavailable.
    """
    if _results_df is None:
        return "Please run the asset mapping first before asking questions about the data."

    if not _AGENT_AVAILABLE:
        return _fallback_answer(user_message)

    try:
        llm = _build_llm()
        # Detect which model is actually being used for accurate metrics
        _model_name = (
            os.getenv("AZURE_ANTHROPIC_DEPLOYMENT", "claude-sonnet-4-6")
            if isinstance(llm, ChatAnthropic)
            else os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
        )
        agent = create_react_agent(llm, _TOOLS, _PROMPT)  # noqa: F821
        executor = AgentExecutor(  # noqa: F821
            agent=agent, tools=_TOOLS,
            verbose=False, handle_parsing_errors=True,
            max_iterations=5,
        )
        from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer
        with _MtTimer() as _t:
            result = executor.invoke({"input": user_message})
        _mt_record(
            model=_model_name,
            purpose="chat_agent", pipeline="chat", node="react_agent",
            success=True, latency_ms=_t.elapsed_ms,
            tokens_in_est=len(user_message) // 4,
        )
        return result.get("output", "I could not generate a response.")
    except Exception:
        return _fallback_answer(user_message)


def _build_data_context() -> str:
    """Build a text summary of the results DataFrame to send as LLM context."""
    if _results_df is None:
        return "No mapping data loaded."
    df = _results_df
    counts  = df["match_type"].value_counts().to_dict()
    avg_c   = round(df["confidence"].mean(), 1)
    unmatched = df[df["match_type"] == "No Match"]["sfm_nav_name"].tolist()[:10]
    low_conf  = df[df["confidence"].between(1, 50)].sort_values("confidence").head(5)[
        ["sfm_nav_name", "confidence", "match_type"]
    ].to_dict(orient="records")
    return (
        f"Total assets: {len(df)}\n"
        f"Match distribution: {counts}\n"
        f"Average confidence: {avg_c}%\n"
        f"Unmatched assets (sample): {unmatched}\n"
        f"Low-confidence assets: {low_conf}"
    )


def _fallback_answer(question: str) -> str:
    """
    Fallback when LangChain agent is unavailable.
    Tries a direct Azure OpenAI GPT-5.5 call with data context first,
    then falls back to keyword matching if the LLM is also unreachable.

    Model: Azure OpenAI GPT-5.5
    Metrics: recorded via llm.metrics_tracker (purpose=chat_direct)
    """
    from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer

    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
    data_ctx = _build_data_context()
    prompt = (
        f"You are a facility management data analyst. "
        f"Here is the current SFM\u2194IFM asset mapping data:\n\n{data_ctx}\n\n"
        f"Answer this question concisely using the data above:\n{question}"
    )

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        with _MtTimer() as _t:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=512,
            )
        answer = resp.choices[0].message.content
        _mt_record(
            model=model, purpose="chat_direct", pipeline="chat",
            node="fallback_llm", success=True, latency_ms=_t.elapsed_ms,
            tokens_in_est=len(prompt) // 4, tokens_out_est=len(answer) // 4,
        )
        return answer
    except Exception:
        _mt_record(model=model, purpose="chat_direct", pipeline="chat",
                   node="fallback_llm", success=False)
        return _keyword_answer(question)


def _keyword_answer(question: str) -> str:
    """Last-resort keyword-based answer when all LLM paths are unavailable."""
    if _results_df is None:
        return "No data available. Please run the asset mapping first."
    df = _results_df
    q  = question.lower()

    if any(w in q for w in ["no match", "unmatched", "not matched"]):
        unmatched = df[df["match_type"] == "No Match"]["sfm_nav_name"].tolist()
        return f"**{len(unmatched)} unmatched assets:**\n" + "\n".join(f"- {a}" for a in unmatched)

    if any(w in q for w in ["summary", "overview", "stats", "total", "how many"]):
        counts = df["match_type"].value_counts()
        lines  = [f"**Total assets:** {len(df)}", f"**Avg confidence:** {round(df['confidence'].mean(),1)}%"]
        for k, v in counts.items():
            lines.append(f"- {k}: {v} ({round(v/len(df)*100,1)}%)")
        return "\n".join(lines)

    if any(w in q for w in ["low confidence", "bad", "problem", "below 50", "worst"]):
        low = df[df["confidence"] < 50].sort_values("confidence").head(10)
        return "**Low confidence assets (< 50%):**\n" + "\n".join(
            f"- {r['sfm_nav_name']}: {r['confidence']}% ({r['match_type']})"
            for _, r in low.iterrows()
        )

    if any(w in q for w in ["building", "site", "campus"]):
        top = df["matched_building"].value_counts().head(5)
        return "**Top buildings by asset count:**\n" + "\n".join(
            f"- {b}: {c} assets" for b, c in top.items()
        )

    return (
        f"I have {len(df)} asset mapping results loaded. "
        "Ask me about: unmatched assets, low confidence scores, building analysis, match type distribution."
    )

