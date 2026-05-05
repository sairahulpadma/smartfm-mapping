"""
langgraph_agent.py
Agentic matching pipeline using LangGraph.

Flow per SFM record:
  START → approach_1 → approach_2 → approach_3
        → partial_1  → partial_2
        → llm_reason → finalize → END

Each approach node either sets match_found=True (and stops) or passes through.
If nothing fires, the LLM reasoning node makes a final call.
"""

from __future__ import annotations

import os
import json
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from rapidfuzz import fuzz

from pipeline.matcher import (
    _score_approach1, _score_approach2, _score_approach3,
    _score_partial1, _score_partial2, find_best_match,
)


# ── State definition ──────────────────────────────────────────────────────────

class AssetMappingState(TypedDict):
    sfm_record: dict
    ifm_records: List[dict]
    candidates: List[dict]          # top candidates so far
    match_found: bool
    match_result: Optional[dict]
    confidence: float
    match_type: str
    approaches_tried: List[str]
    reasoning: str
    needs_llm: bool


# ── Approach nodes ────────────────────────────────────────────────────────────

def _run_approach(state: AssetMappingState, name: str, scorer) -> AssetMappingState:
    if state["match_found"]:
        return state

    sfm = state["sfm_record"]
    best_score, best_ifm = 0.0, None

    for ifm in state["ifm_records"]:
        score = scorer(sfm, ifm)
        if score is not None and score > best_score:
            best_score, best_ifm = score, ifm

    tried = state["approaches_tried"] + [name]

    if best_ifm and best_score > 0:
        result = {
            "sfm_nav_name": sfm.get("nav_name", ""),
            "matched_asset_id": best_ifm.get("asset_id", ""),
            "matched_asset_name": best_ifm.get("asset_name", ""),
            "matched_position_name": best_ifm.get("position_name", ""),
            "matched_building": best_ifm.get("building_name", ""),
            "match_type": name,
            "approach_used": name,
            "confidence": round(best_score * 100, 1),
        }
        return {**state, "match_found": True, "match_result": result,
                "confidence": round(best_score * 100, 1),
                "match_type": name, "approaches_tried": tried}

    return {**state, "approaches_tried": tried}


def node_approach1(state: AssetMappingState) -> AssetMappingState:
    return _run_approach(state, "Perfect - Approach 1", _score_approach1)

def node_approach2(state: AssetMappingState) -> AssetMappingState:
    return _run_approach(state, "Perfect - Approach 2", _score_approach2)

def node_approach3(state: AssetMappingState) -> AssetMappingState:
    return _run_approach(state, "Perfect - Approach 3", _score_approach3)

def node_partial1(state: AssetMappingState) -> AssetMappingState:
    return _run_approach(state, "Partial - Approach 1", _score_partial1)

def node_partial2(state: AssetMappingState) -> AssetMappingState:
    return _run_approach(state, "Partial - Approach 2", _score_partial2)


# ── LLM reasoning node ────────────────────────────────────────────────────────

def node_llm_reason(state: AssetMappingState) -> AssetMappingState:
    """
    For truly ambiguous records, ask the LLM to make a final decision.
    Falls back gracefully if LLM is unavailable.
    """
    if state["match_found"]:
        return state

    sfm = state["sfm_record"]

    # Get top-5 IFM candidates by raw name similarity for LLM context
    from rapidfuzz import fuzz as _fuzz
    scored = []
    nav = str(sfm.get("nav_name", ""))
    for ifm in state["ifm_records"]:
        s = max(
            _fuzz.token_set_ratio(nav, str(ifm.get("asset_name", ""))),
            _fuzz.token_set_ratio(nav, str(ifm.get("position_name", ""))),
        )
        scored.append((s, ifm))
    top5 = [ifm for _, ifm in sorted(scored, key=lambda x: -x[0])[:5]]

    # Build LLM prompt
    candidates_text = json.dumps(
        [{k: ifm.get(k, "") for k in
          ["asset_id", "asset_name", "position_name", "manufacturer",
           "serial_number", "model", "building_name", "region_name"]}
         for ifm in top5],
        indent=2
    )

    prompt = f"""You are a facility management data expert matching assets between two systems.

SFM Asset (source):
{json.dumps({k: sfm.get(k, "") for k in ["nav_name", "equip_type", "equip_make", "equip_model", "equip_serial", "country", "state", "city", "site_name"]}, indent=2)}

Top IFM candidates (target):
{candidates_text}

Task: Decide if any IFM candidate is a match for the SFM asset.
Respond ONLY with valid JSON:
{{
  "match": true/false,
  "asset_id": "<matched asset_id or null>",
  "asset_name": "<matched asset_name or null>",
  "confidence": <0-100>,
  "reasoning": "<one sentence explanation>"
}}"""

    try:
        # Try Azure OpenAI first
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        resp = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1"),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=512,
            response_format={"type": "json_object"},
        )
        llm_out = json.loads(resp.choices[0].message.content)
    except Exception:
        # Fallback: no match
        llm_out = {"match": False, "asset_id": None, "asset_name": None,
                   "confidence": 0, "reasoning": "LLM unavailable – no match determined"}

    if llm_out.get("match") and llm_out.get("asset_id"):
        result = {
            "sfm_nav_name": sfm.get("nav_name", ""),
            "matched_asset_id": llm_out.get("asset_id"),
            "matched_asset_name": llm_out.get("asset_name"),
            "matched_position_name": "",
            "matched_building": "",
            "match_type": "LLM Reasoned",
            "approach_used": "LLM Reasoning",
            "confidence": float(llm_out.get("confidence", 50)),
        }
        return {**state, "match_found": True, "match_result": result,
                "confidence": float(llm_out.get("confidence", 50)),
                "match_type": "LLM Reasoned",
                "reasoning": llm_out.get("reasoning", ""),
                "approaches_tried": state["approaches_tried"] + ["llm_reason"]}

    return {**state,
            "reasoning": llm_out.get("reasoning", "No suitable match found"),
            "approaches_tried": state["approaches_tried"] + ["llm_reason"]}


# ── Finalize node ─────────────────────────────────────────────────────────────

def node_finalize(state: AssetMappingState) -> AssetMappingState:
    if not state["match_found"]:
        sfm = state["sfm_record"]
        result = {
            "sfm_nav_name": sfm.get("nav_name", ""),
            "matched_asset_id": None,
            "matched_asset_name": None,
            "matched_position_name": None,
            "matched_building": None,
            "match_type": "No Match",
            "approach_used": "None",
            "confidence": 0.0,
        }
        return {**state, "match_result": result, "match_type": "No Match"}
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_approach(state: AssetMappingState) -> str:
    return END if state["match_found"] else "continue"


# ── Build the graph ───────────────────────────────────────────────────────────

def _make_router(next_node: str):
    """Return a routing function that goes to next_node unless match already found."""
    def router(state: AssetMappingState) -> str:
        return END if state["match_found"] else next_node
    return router


def build_graph() -> StateGraph:
    g = StateGraph(AssetMappingState)

    g.add_node("approach1",  node_approach1)
    g.add_node("approach2",  node_approach2)
    g.add_node("approach3",  node_approach3)
    g.add_node("partial1",   node_partial1)
    g.add_node("partial2",   node_partial2)
    g.add_node("llm_reason", node_llm_reason)
    g.add_node("finalize",   node_finalize)

    g.set_entry_point("approach1")

    edges = [
        ("approach1", "approach2"),
        ("approach2", "approach3"),
        ("approach3", "partial1"),
        ("partial1",  "partial2"),
        ("partial2",  "llm_reason"),
        ("llm_reason","finalize"),
    ]
    for src, dst in edges:
        g.add_conditional_edges(src, _make_router(dst), {END: END, dst: dst})

    g.add_edge("finalize", END)
    return g.compile()


# ── Public runner ─────────────────────────────────────────────────────────────

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_single(sfm_record: dict, ifm_records: list[dict]) -> dict:
    """Run the LangGraph pipeline for a single SFM record."""
    initial_state: AssetMappingState = {
        "sfm_record": sfm_record,
        "ifm_records": ifm_records,
        "candidates": [],
        "match_found": False,
        "match_result": None,
        "confidence": 0.0,
        "match_type": "No Match",
        "approaches_tried": [],
        "reasoning": "",
        "needs_llm": False,
    }
    final = get_graph().invoke(initial_state)
    return final["match_result"]


def run_pipeline(sfm_records: list[dict], ifm_records: list[dict],
                 progress_callback=None) -> list[dict]:
    """
    Run the full mapping pipeline for all SFM records.
    progress_callback(i, total) called after each record if provided.
    """
    results = []
    total = len(sfm_records)
    for i, sfm in enumerate(sfm_records):
        result = run_single(sfm, ifm_records)
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, total)
    return results
