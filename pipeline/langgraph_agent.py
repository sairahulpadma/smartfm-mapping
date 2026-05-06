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
import logging
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from rapidfuzz import fuzz

from pipeline.matcher import (
    _score_approach1, _score_approach2, _score_approach3,
    _score_partial1, _score_partial2, find_best_match,
)


logger = logging.getLogger(__name__)

# ── State definition ────────────────────────────────────────────────────────────────────────

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
    llm_verify_note: str            # set by node_llm_verify_partial


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


# ── LLM partial-verification node ──────────────────────────────────────────────────

def node_llm_verify_partial(state: AssetMappingState) -> AssetMappingState:
    """
    LLM Verification Node — runs after partial1 / partial2 when a fuzzy
    partial match is found (50–84 % confidence).

    Asks Azure OpenAI GPT-5.5 to semantically verify whether the fuzzy
    partial match is genuinely correct.  The LLM can:
      • Confirm the match → match_type becomes "<Approach> (LLM Verified)"
        and confidence may be adjusted upward.
      • Reject the match → match_found is cleared so node_llm_reason
        gets a full attempt from scratch.

    Model used : Azure OpenAI GPT-5.5 (deployment: gpt-5.5_1)
    Metrics    : recorded via llm.metrics_tracker (purpose=asset_matching_partial)
    """
    # Only verify when a partial match exists
    if not state["match_found"] or "Partial" not in state.get("match_type", ""):
        return state

    match = state["match_result"]
    sfm   = state["sfm_record"]
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")

    # Compact prompt — saves ~40 tokens vs verbose version
    prompt = (
        f"Verify FM asset partial match ({state['confidence']:.0f}% fuzzy).\n"
        f"SFM: name={sfm.get('nav_name','')} type={sfm.get('equip_type','')} "
        f"make={sfm.get('equip_make','')} model={sfm.get('equip_model','')} "
        f"city={sfm.get('city','')} site={sfm.get('site_name','')}\n"
        f"IFM: name={match.get('matched_asset_name','')} "
        f"pos={match.get('matched_position_name','')} "
        f"building={match.get('matched_building','')}\n"
        f'Is this the same physical asset? JSON only: {{"confirmed":true/false,'
        f'"adjusted_confidence":<0-100>,"reasoning":"<1 sentence>"}}'
    )

    try:
        from openai import AzureOpenAI
        from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        with _MtTimer() as _t:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                response_format={"type": "json_object"},
            )
        raw = resp.choices[0].message.content
        llm_out = json.loads(raw)

        confirmed    = bool(llm_out.get("confirmed", True))
        new_conf     = float(llm_out.get("adjusted_confidence", state["confidence"]))
        reasoning    = llm_out.get("reasoning", "")
        tried        = state["approaches_tried"] + ["llm_verify_partial"]

        _mt_record(
            model=model, purpose="asset_matching_partial",
            pipeline="asset_mapping", node="llm_verify_partial",
            success=True, latency_ms=_t.elapsed_ms,
            tokens_in_est=len(prompt) // 4,
            tokens_out_est=len(raw) // 4,
            extra={"confirmed": confirmed, "new_conf": new_conf},
        )
        logger.info(
            "[LLM] Partial verify → confirmed=%s conf=%.1f%% in %dms  model=%s",
            confirmed, new_conf, _t.elapsed_ms, model,
        )

        if confirmed:
            new_type = f"{state['match_type']} (LLM Verified)"
            updated  = {**match, "match_type": new_type, "confidence": new_conf}
            return {**state,
                    "match_result":    updated,
                    "confidence":      new_conf,
                    "match_type":      new_type,
                    "reasoning":       reasoning,
                    "llm_verify_note": reasoning,
                    "approaches_tried": tried}
        else:
            # Reject partial — let node_llm_reason try from scratch
            return {**state,
                    "match_found":     False,
                    "match_result":    None,
                    "llm_verify_note": f"LLM rejected partial: {reasoning}",
                    "approaches_tried": tried}

    except Exception as exc:
        logger.warning("[LLM] llm_verify_partial failed (%s) — keeping partial as-is", exc)
        try:
            from llm.metrics_tracker import record as _mt_record
            _mt_record(model=model, purpose="asset_matching_partial",
                       pipeline="asset_mapping", node="llm_verify_partial", success=False)
        except Exception:
            pass
        return state  # graceful fallback: keep the partial match unchanged

# ── LLM reasoning node ────────────────────────────────────────────────────────────────────────

def node_llm_reason(state: AssetMappingState) -> AssetMappingState:
    """
    Final LLM reasoning node — handles truly ambiguous records where all
    5 fuzzy approaches and the partial verifier failed to find a match.

    Sends top-5 IFM candidates (by raw name similarity) to GPT-5.5 and
    asks for a match decision with reasoning.

    Model used : Azure OpenAI GPT-5.5 (deployment: gpt-5.5_1)
    Metrics    : recorded via llm.metrics_tracker (purpose=asset_matching_reason)
    Falls back : gracefully returns no-match if LLM is unavailable.
    """
    if state["match_found"]:
        return state

    sfm = state["sfm_record"]

    # Get top-3 IFM candidates (3 is enough context, saves ~30% tokens vs top-5)
    from rapidfuzz import fuzz as _fuzz
    scored = []
    nav = str(sfm.get("nav_name", ""))
    for ifm in state["ifm_records"]:
        s = max(
            _fuzz.token_set_ratio(nav, str(ifm.get("asset_name", ""))),
            _fuzz.token_set_ratio(nav, str(ifm.get("position_name", ""))),
        )
        scored.append((s, ifm))
    top3 = [ifm for _, ifm in sorted(scored, key=lambda x: -x[0])[:3]]

    # Compact JSON — no indent, only essential keys
    _sfm_ctx = {k: sfm.get(k, "") for k in
                ["nav_name", "equip_type", "equip_make", "equip_model",
                 "equip_serial", "city", "state", "site_name"]}
    _cands   = [{k: ifm.get(k, "") for k in
                 ["asset_id", "asset_name", "position_name",
                  "manufacturer", "model", "building_name"]}
                for ifm in top3]

    prompt = (
        f"FM asset matching. Decide if any IFM candidate matches the SFM asset.\n"
        f"SFM: {json.dumps(_sfm_ctx, separators=(',', ':'))}\n"
        f"IFM candidates: {json.dumps(_cands, separators=(',', ':'))}\n"
        f'Respond ONLY with JSON: {{"match":true/false,"asset_id":"<id or null>",'
        f'"asset_name":"<name or null>","confidence":<0-100>,"reasoning":"<1 sentence>"}}'
    )

    try:
        # Routing: Claude Sonnet 4.6 for nuanced multi-candidate reasoning (saves GPT quota)
        # Fallback: GPT-5.5 if Anthropic creds not set
        from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer
        from anthropic import AnthropicFoundry as _AnthropicFoundry
        import httpx as _httpx

        _ant_key      = os.getenv("AZURE_ANTHROPIC_API_KEY", "")
        _ant_endpoint = os.getenv("AZURE_ANTHROPIC_ENDPOINT",
                                  "https://admv-mogidbp0-eastus2.services.ai.azure.com/anthropic/")
        _ant_model    = os.getenv("AZURE_ANTHROPIC_DEPLOYMENT", "claude-sonnet-4-6")

        # Skip if key is missing or still the placeholder value
        if _ant_key and not _ant_key.startswith("<"):
            ant_client = _AnthropicFoundry(
                api_key=_ant_key,
                base_url=_ant_endpoint,
                http_client=_httpx.Client(verify=False),
            )
            with _MtTimer() as _t:
                msg = ant_client.messages.create(
                    model=_ant_model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
            raw_text = msg.content[0].text
            llm_out  = json.loads(raw_text)
            _model   = _ant_model
        else:
            raise RuntimeError("No valid Anthropic key — fall through to GPT")

    except Exception:
        # Fallback to GPT-5.5
        try:
            from openai import AzureOpenAI
            from llm.metrics_tracker import record as _mt_record, Timer as _MtTimer

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
            _model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
            raw_resp = None
            with _MtTimer() as _t:
                resp = client.chat.completions.create(
                    model=_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=256,
                    response_format={"type": "json_object"},
                )
            raw_text = resp.choices[0].message.content
            llm_out  = json.loads(raw_text)
        except Exception:
            try:
                from llm.metrics_tracker import record as _mt_record
                _mt_record(model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1"),
                           purpose="asset_matching_reason", pipeline="asset_mapping",
                           node="llm_reason", success=False)
            except Exception:
                pass
            llm_out = {"match": False, "asset_id": None, "asset_name": None,
                       "confidence": 0, "reasoning": "LLM unavailable – no match determined"}
            _model  = "unknown"
            _t      = type("T", (), {"elapsed_ms": 0})()
            raw_text = ""

    try:
        _mt_record(
            model=_model, purpose="asset_matching_reason",
            pipeline="asset_mapping", node="llm_reason",
            success=True, latency_ms=_t.elapsed_ms,
            tokens_in_est=len(prompt) // 4,
            tokens_out_est=len(raw_text) // 4,
            extra={"matched": bool(llm_out.get("match")),
                   "confidence": llm_out.get("confidence", 0)},
        )
    except Exception:
        pass
    logger.info(
        "[LLM] Asset reason → match=%s conf=%s in %dms  model=%s",
        llm_out.get("match"), llm_out.get("confidence"),
        getattr(_t, "elapsed_ms", 0), _model,
    )

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
    """
    Build the 8-node LangGraph asset-matching pipeline.

    Flow:
      START → approach1 → approach2 → approach3
            → partial1 → partial2
            → llm_verify_partial   (NEW: verify fuzzy partial matches with GPT-5.5)
            → llm_reason           (full LLM reasoning for unresolved records)
            → finalize → END

    Early exits: each approach/partial node exits to END on match_found=True,
    EXCEPT partial nodes route to llm_verify_partial first for LLM confirmation.
    """
    g = StateGraph(AssetMappingState)

    g.add_node("approach1",          node_approach1)
    g.add_node("approach2",          node_approach2)
    g.add_node("approach3",          node_approach3)
    g.add_node("partial1",           node_partial1)
    g.add_node("partial2",           node_partial2)
    g.add_node("llm_verify_partial", node_llm_verify_partial)  # NEW
    g.add_node("llm_reason",         node_llm_reason)
    g.add_node("finalize",           node_finalize)

    g.set_entry_point("approach1")

    # Perfect approaches: stop immediately on match_found
    for src, dst in [("approach1", "approach2"), ("approach2", "approach3"),
                     ("approach3", "partial1")]:
        g.add_conditional_edges(src, _make_router(dst), {END: END, dst: dst})

    # Partial approaches: route to LLM verifier on match_found
    g.add_conditional_edges(
        "partial1",
        lambda s: "llm_verify_partial" if s["match_found"] else "partial2",
        {"llm_verify_partial": "llm_verify_partial", "partial2": "partial2"},
    )
    g.add_conditional_edges(
        "partial2",
        lambda s: "llm_verify_partial" if s["match_found"] else "llm_reason",
        {"llm_verify_partial": "llm_verify_partial", "llm_reason": "llm_reason"},
    )

    # After LLM verify: confirmed → finalize, rejected → llm_reason
    g.add_conditional_edges(
        "llm_verify_partial",
        lambda s: "finalize" if s["match_found"] else "llm_reason",
        {"finalize": "finalize", "llm_reason": "llm_reason"},
    )

    g.add_conditional_edges(
        "llm_reason", _make_router("finalize"), {END: END, "finalize": "finalize"}
    )
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
        "sfm_record":       sfm_record,
        "ifm_records":      ifm_records,
        "candidates":       [],
        "match_found":      False,
        "match_result":     None,
        "confidence":       0.0,
        "match_type":       "No Match",
        "approaches_tried": [],
        "reasoning":        "",
        "needs_llm":        False,
        "llm_verify_note":  "",
    }
    final = get_graph().invoke(initial_state)
    result = final["match_result"] or {}
    # Attach the LangGraph nodes visited so the UI can show the full cascade
    result["approaches_tried"] = " → ".join(final.get("approaches_tried", []))
    result["reasoning"] = final.get("reasoning", "")
    return result


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
