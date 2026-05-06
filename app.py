"""
app.py  –  SFM ↔ IFM Asset Mapping Platform
Streamlit dashboard with:
  Tab 1 – Upload & Map (file upload → LangGraph pipeline → results table)
  Tab 2 – Dashboard   (charts, KPIs, building analysis)
  Tab 3 – AI Chat     (LangChain agent for natural language queries)
"""

import os
import io
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SFM ↔ IFM AI Mapping",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border: 1px solid #0f3460;
    border-radius: 12px; padding: 18px 24px;
    color: white; text-align: center; margin-bottom: 8px;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; margin: 4px 0; }
  .kpi-label { font-size: 0.85rem; color: #aab; }
  .perfect  { color: #00e676; }
  .partial  { color: #ffca28; }
  .llm      { color: #40c4ff; }
  .nomatch  { color: #ff5252; }
  .chat-user { background:#1e293b; border-radius:8px; padding:10px 14px; margin:4px 0; }
  .chat-bot  { background:#0f3460; border-radius:8px; padding:10px 14px; margin:4px 0; }
  div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("results_df", None),
    ("sfm_df", None),
    ("ifm_df", None),
    ("chat_history", []),
    ("mapping_done", False),
    ("sensor_outcomes", []),
    ("pipeline_obj", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/building.png", width=60)
    st.title("SFM ↔ IFM Platform")
    st.caption("Gen AI  •  LangGraph  •  LangChain")
    st.divider()

    st.subheader("⚙️ Settings")
    use_llm = st.toggle("Enable LLM Reasoning", value=True,
                        help="Uses GPT-5.5 for ambiguous matches")
    show_debug = st.toggle("Show Debug Info", value=False)

    st.divider()
    st.markdown("""
**Pipeline:**  
`Upload → LangGraph → Scorer → LLM`

**Match Levels:**  
🟢 Perfect Match (≥85 %)  
🟡 Partial Match (50–84 %)  
🔵 LLM Reasoned (AI decided)  
🔴 No Match (0 %)
""")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏢 SFM ↔ IFM Asset Mapping Platform")
st.caption("Powered by **Gen AI** | **LangGraph** | **Azure OpenAI GPT-5.5** | **Claude Sonnet 4.6**")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂  Upload & Map",
    "📊  Dashboard",
    "💬  AI Assistant",
    "🌡️  Sensor Monitor",
    "🔍  Review Queue",
])


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 1 – Upload & Map
# ╚══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Step 1 — Upload Your Data Files")

    col_sfm, col_ifm = st.columns(2)
    with col_sfm:
        sfm_file = st.file_uploader(
            "📤 Upload SFM Excel", type=["xlsx", "xls"],
            key="sfm_upload", help="Smart FM export file"
        )
    with col_ifm:
        ifm_file = st.file_uploader(
            "📤 Upload IFM Excel", type=["xlsx", "xls"],
            key="ifm_upload", help="IFM Hub export file"
        )

    # Option B – use the combined hackathon file directly
    st.divider()
    st.subheader("OR – Use the Combined Hackathon File")
    combined_file = st.file_uploader(
        "📤 Upload Combined SFM+IFM Excel (hackathon format)",
        type=["xlsx", "xls"], key="combined_upload",
    )

    st.divider()
    # ── Demo mode: load pre-computed results instantly ────────────────────────
    demo_path = "data/test/demo_results.csv"
    import os as _os
    if _os.path.exists(demo_path):
        if st.button("🎯 Load Demo Results (all 4 match tiers — no upload needed)",
                     type="secondary"):
            demo_df = pd.read_csv(demo_path)
            st.session_state.results_df = demo_df
            st.session_state.mapping_done = True
            from llm.chat_agent import set_results
            set_results(demo_df)
            st.success("✅ Demo results loaded! Switch to the **Dashboard** or **AI Assistant** tab.")
            st.rerun()

    st.divider()
    run_btn = st.button("🚀 Run AI Mapping Pipeline", type="primary",
                        disabled=(sfm_file is None and ifm_file is None
                                  and combined_file is None))

    if run_btn:
        from pipeline.data_loader import load_sfm_ifm_from_excel, load_separate_files, records_to_dicts
        from pipeline.langgraph_agent import run_pipeline

        with st.spinner("Loading data..."):
            if combined_file:
                raw_bytes = combined_file.read()
                sfm_df, ifm_df = load_sfm_ifm_from_excel(io.BytesIO(raw_bytes))
            else:
                sfm_bytes = sfm_file.read()
                ifm_bytes = ifm_file.read()
                sfm_df, ifm_df = load_separate_files(
                    io.BytesIO(sfm_bytes), io.BytesIO(ifm_bytes)
                )
            st.session_state.sfm_df = sfm_df
            st.session_state.ifm_df = ifm_df

        sfm_records = records_to_dicts(sfm_df)
        ifm_records = records_to_dicts(ifm_df)
        total = len(sfm_records)

        st.info(f"Loaded **{total} SFM records** and **{len(ifm_records)} IFM records**. Running pipeline…")

        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        def on_progress(i, t):
            progress_bar.progress(i / t)
            status_text.text(f"Processing {i}/{t} assets…")

        with st.spinner("Running LangGraph matching pipeline…"):
            if use_llm:
                results = run_pipeline(sfm_records, ifm_records, progress_callback=on_progress)
            else:
                from pipeline.matcher import run_bulk_matching
                results = run_bulk_matching(sfm_records, ifm_records)
                progress_bar.progress(1.0)

        results_df = pd.DataFrame(results)
        st.session_state.results_df = results_df
        st.session_state.mapping_done = True

        # Inject into chat agent
        from llm.chat_agent import set_results
        set_results(results_df)

        status_text.success(f"✅ Mapping complete! {total} assets processed.")
        st.balloons()

    # ── Results table ─────────────────────────────────────────────────────────
    if st.session_state.mapping_done and st.session_state.results_df is not None:
        st.divider()
        st.subheader("📋 Mapping Results")

        df = st.session_state.results_df.copy()

        # Color-coded match type
        def color_match(val):
            colors = {
                "No Match": "background-color:#3d0000;color:#ff5252",
                "LLM Reasoned": "background-color:#001f3d;color:#40c4ff",
            }
            if "Perfect" in str(val):
                return "background-color:#003d1a;color:#00e676"
            if "Partial" in str(val):
                return "background-color:#3d2f00;color:#ffca28"
            return colors.get(val, "")

        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            match_filter = st.multiselect(
                "Filter by Match Type",
                options=df["match_type"].unique().tolist(),
                default=df["match_type"].unique().tolist(),
            )
        with fc2:
            conf_min = st.slider("Min Confidence %", 0, 100, 0)
        with fc3:
            search = st.text_input("Search asset name")

        filtered = df[df["match_type"].isin(match_filter)]
        filtered = filtered[filtered["confidence"] >= conf_min]
        if search:
            filtered = filtered[
                filtered["sfm_nav_name"].str.contains(search, case=False, na=False)
            ]

        styled = filtered.style.map(color_match, subset=["match_type"])
        st.dataframe(styled, use_container_width=True, height=420)

        # Download
        csv = filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv,
            file_name="sfm_ifm_mapping_results.csv",
            mime="text/csv",
        )

        if show_debug:
            with st.expander("🔍 Raw SFM Data"):
                st.dataframe(st.session_state.sfm_df, use_container_width=True)
            with st.expander("🔍 Raw IFM Data"):
                st.dataframe(st.session_state.ifm_df, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 2 – Dashboard
# ╚══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.mapping_done or st.session_state.results_df is None:
        st.info("👆 Run the mapping pipeline in the **Upload & Map** tab first.")
    else:
        df = st.session_state.results_df.copy()
        total = len(df)
        perfect = df["match_type"].str.contains("Perfect", na=False).sum()
        partial = df["match_type"].str.contains("Partial", na=False).sum()
        llm_match = (df["match_type"] == "LLM Reasoned").sum()
        no_match = (df["match_type"] == "No Match").sum()
        avg_conf = round(df["confidence"].mean(), 1)

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.subheader("📈 Key Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        cards = [
            (k1, "Total Assets", total, ""),
            (k2, "Perfect Matches", f"{perfect} ({round(perfect/total*100,1)}%)", "perfect"),
            (k3, "Partial Matches", f"{partial} ({round(partial/total*100,1)}%)", "partial"),
            (k4, "LLM Reasoned",   f"{llm_match} ({round(llm_match/total*100,1)}%)", "llm"),
            (k5, "No Match",       f"{no_match} ({round(no_match/total*100,1)}%)", "nomatch"),
        ]
        for col, label, value, cls in cards:
            with col:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value {cls}">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Charts row 1 ──────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            st.subheader("Match Type Distribution")
            counts = df["match_type"].value_counts().reset_index()
            counts.columns = ["Match Type", "Count"]
            color_map = {
                "Perfect - Approach 1": "#00e676",
                "Perfect - Approach 2": "#69f0ae",
                "Perfect - Approach 3": "#b9f6ca",
                "Partial - Approach 1": "#ffca28",
                "Partial - Approach 2": "#ffd740",
                "LLM Reasoned":         "#40c4ff",
                "No Match":             "#ff5252",
            }
            fig = px.pie(
                counts, names="Match Type", values="Count",
                color="Match Type", color_discrete_map=color_map,
                hole=0.45,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.subheader("Confidence Score Distribution")
            fig2 = px.histogram(
                df, x="confidence", nbins=20,
                color_discrete_sequence=["#40c4ff"],
                labels={"confidence": "Confidence %", "count": "Assets"},
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                               plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(gridcolor="#1a1a2e"),
                               yaxis=dict(gridcolor="#1a1a2e"))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Charts row 2 ──────────────────────────────────────────────────────
        ch3, ch4 = st.columns(2)

        with ch3:
            st.subheader("Match Quality by Building")
            bldg = df.copy()
            bldg["building"] = bldg["matched_building"].fillna("Unknown").replace("", "Unknown")
            bldg_grp = bldg.groupby("building").apply(
                lambda x: pd.Series({
                    "Perfect": x["match_type"].str.contains("Perfect", na=False).sum(),
                    "Partial": x["match_type"].str.contains("Partial", na=False).sum(),
                    "No Match": (x["match_type"] == "No Match").sum(),
                })
            ).reset_index()
            bldg_melt = bldg_grp.melt(id_vars="building",
                                       value_vars=["Perfect", "Partial", "No Match"],
                                       var_name="Match Type", value_name="Count")
            fig3 = px.bar(
                bldg_melt, x="building", y="Count", color="Match Type",
                color_discrete_map={"Perfect": "#00e676", "Partial": "#ffca28",
                                     "No Match": "#ff5252"},
                barmode="stack",
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                               plot_bgcolor="rgba(0,0,0,0)",
                               xaxis_tickangle=-30,
                               xaxis=dict(gridcolor="#1a1a2e"),
                               yaxis=dict(gridcolor="#1a1a2e"))
            st.plotly_chart(fig3, use_container_width=True)

        with ch4:
            st.subheader("Top 10 Lowest Confidence Assets")
            low = df[df["confidence"] > 0].nsmallest(10, "confidence")[
                ["sfm_nav_name", "confidence", "match_type"]
            ]
            fig4 = px.bar(
                low, x="confidence", y="sfm_nav_name", orientation="h",
                color="confidence",
                color_continuous_scale=["#ff5252", "#ffca28", "#00e676"],
                labels={"sfm_nav_name": "Asset", "confidence": "Confidence %"},
            )
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                               plot_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(gridcolor="#1a1a2e"),
                               xaxis=dict(gridcolor="#1a1a2e"))
            st.plotly_chart(fig4, use_container_width=True)

        # ── Unmatched assets table ─────────────────────────────────────────────
        st.divider()
        st.subheader("🔴 Unmatched Assets — Require Manual Review")
        unmatched = df[df["match_type"] == "No Match"][["sfm_nav_name", "confidence"]]
        if unmatched.empty:
            st.success("🎉 All assets matched successfully!")
        else:
            st.warning(f"⚠️ {len(unmatched)} assets could not be matched automatically.")
            st.dataframe(unmatched, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 3 – AI Chat Assistant
# ╚══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💬 AI Assistant — Ask Anything About Your Mapping Data")
    st.caption("Powered by LangChain + Claude Sonnet 4.6 / GPT-5.5")

    if not st.session_state.mapping_done:
        st.info("👆 Run the mapping pipeline in the **Upload & Map** tab first.")
    else:
        # Inject current results into agent
        from llm.chat_agent import set_results, get_chat_response
        if st.session_state.results_df is not None:
            set_results(st.session_state.results_df)

        # Suggested prompts
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "Which assets have no match?",
            "Give me a summary of mapping quality",
            "Show me assets with confidence below 50%",
            "Which building has the most unmatched assets?",
            "What percentage of assets matched perfectly?",
        ]
        sugg_cols = st.columns(len(suggestions))
        for i, (col, s) in enumerate(zip(sugg_cols, suggestions)):
            with col:
                if st.button(s, key=f"sugg_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    with st.spinner("Thinking…"):
                        reply = get_chat_response(s, st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

        st.divider()

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask about your data… (e.g. 'which assets are unmatched?')")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data…"):
                    reply = get_chat_response(user_input, st.session_state.chat_history)
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 4 – Sensor Monitor
# ╚══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🌡️ Sensor Event Monitor → Work Order Pipeline")
    st.caption(
        "Ingest sensor alerts, classify service type, and auto-create or queue work orders in IFM Hub."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("""
**Pipeline flow:**
```
Sensor Alert  →  Service Classifier (4-tier)  →  Decision Engine
                                                    ├─ Perfect Match  → ✅ Auto-Create Work Order in IFM Hub
                                                    ├─ Partial Match  → ⚠️  Enqueue for Human Review
                                                    ├─ LLM Reasoned  → 🤖 Enqueue for Human Review
                                                    └─ No Match      → ❌ Log Only (no request)
```
""")
    with col_b:
        demo_mode = st.toggle("Demo Mode (no real API)", value=True)
        use_llm_sensor = st.toggle("LLM Fallback", value=True)

    st.divider()

    # ── Load demo sensor events ───────────────────────────────────────────────
    col_run, col_clear = st.columns([3, 1])
    with col_run:
        if st.button("🚀 Run Demo Sensor Events", type="primary"):
            from pipeline.sensor_ingestor import make_demo_events
            from pipeline.orchestrator import get_pipeline, reset_pipeline

            reset_pipeline()
            pipe = get_pipeline(use_llm=use_llm_sensor, demo_mode=demo_mode)
            st.session_state.pipeline_obj = pipe

            demo_events = make_demo_events()
            with st.spinner(f"Processing {len(demo_events)} sensor events…"):
                outcomes = pipe.process_batch(demo_events)
            st.session_state.sensor_outcomes = outcomes
            st.success(f"✅ Processed {len(outcomes)} sensor events.")
            st.rerun()

    with col_clear:
        if st.button("🗑️ Clear"):
            st.session_state.sensor_outcomes = []
            st.rerun()

    # ── Custom sensor event form ──────────────────────────────────────────────
    with st.expander("➕ Simulate a Custom Sensor Alert"):
        f1, f2, f3 = st.columns(3)
        with f1:
            c_asset_type  = st.selectbox("Asset Type", [
                "AHU", "FCU", "Chiller", "Thermostat", "Electrical Panel",
                "Generator", "UPS", "Water Leak Sensor", "Pump", "Freezer",
                "BMS", "Fire Alarm Panel", "Elevator", "CCTV Camera",
            ])
            c_alert_type = st.selectbox("Alert Type", [
                "temperature_high", "temperature_low", "cooling_failure",
                "heating_failure", "thermostat_fault", "setpoint_deviation",
                "power_failure", "breaker_trip", "voltage_drop",
                "water_leak", "flood_detection", "vibration_high",
                "pump_failure", "freezer_temp_high", "fire_alarm",
                "elevator_fault", "camera_offline", "controller_offline",
                "energy_spike",
            ])
        with f2:
            c_asset_name = st.text_input("Asset Name", value="AHU-1 Level B1")
            c_building   = st.text_input("Building", value="HQ Building A")
        with f3:
            c_reading_val  = st.number_input("Reading Value", value=85.0)
            c_reading_unit = st.text_input("Unit", value="°F")
            c_threshold    = st.number_input("Max Threshold", value=72.0)

        if st.button("⚡ Submit Custom Sensor Alert"):
            from pipeline.orchestrator import get_pipeline
            pipe = get_pipeline(use_llm=use_llm_sensor, demo_mode=demo_mode)
            raw = {
                "sensor_id":  f"SNS-CUSTOM-{c_asset_type.replace(' ', '-').upper()}",
                "asset_id":   f"SFM-CUSTOM-001",
                "asset_name": c_asset_name,
                "asset_type": c_asset_type,
                "alert_type": c_alert_type,
                "building":   c_building,
                "floor":      "1",
                "room":       "Custom Room",
                "location_id": "f754334d-17cc-4890-bc58-2a4e1a386549",
                "reading": {
                    "parameter": "custom",
                    "value": c_reading_val,
                    "unit": c_reading_unit,
                    "threshold_max": c_threshold,
                },
            }
            outcome = pipe.process_raw_event(raw)
            st.session_state.sensor_outcomes.append(outcome)
            st.success(f"Event processed! Decision: **{outcome['decision']}** | Match: **{outcome['match_type']}** | Confidence: **{outcome.get('confidence', 0):.1f}%**")
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.sensor_outcomes:
        outcomes = st.session_state.sensor_outcomes
        df_out = pd.DataFrame(outcomes)

        st.divider()
        st.subheader("📊 Pipeline Results")

        # KPIs
        total_s = len(outcomes)
        auto    = sum(1 for o in outcomes if o.get("decision") == "AUTO_CREATE")
        review  = sum(1 for o in outcomes if o.get("decision") == "REVIEW")
        no_act  = sum(1 for o in outcomes if o.get("decision") == "NO_ACTION")

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Events Processed</div><div class="kpi-value">{total_s}</div></div>', unsafe_allow_html=True)
        with kc2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">✅ Auto-Created</div><div class="kpi-value perfect">{auto}</div></div>', unsafe_allow_html=True)
        with kc3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">⚠️ Pending Review</div><div class="kpi-value partial">{review}</div></div>', unsafe_allow_html=True)
        with kc4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">❌ No Action</div><div class="kpi-value nomatch">{no_act}</div></div>', unsafe_allow_html=True)

        st.divider()

        # Color-code decision column
        def _color_decision(val):
            if val == "AUTO_CREATE": return "background-color:#003d1a;color:#00e676"
            if val == "REVIEW":      return "background-color:#3d2f00;color:#ffca28"
            return "background-color:#3d0000;color:#ff5252"

        cols_show = [c for c in [
            "asset_name", "alert_type", "severity", "building",
            "service_classification_name", "match_type", "confidence", "decision",
        ] if c in df_out.columns]

        st.dataframe(
            df_out[cols_show].style.map(_color_decision, subset=["decision"]),
            use_container_width=True, height=360,
        )

        # Decision chart
        _dec_counts = df_out["decision"].value_counts().reset_index()
        _dec_counts.columns = ["Decision", "Count"]
        fig_d = px.pie(
            _dec_counts,
            names="Decision", values="Count",
            color="Decision",
            color_discrete_map={
                "AUTO_CREATE": "#00e676",
                "REVIEW":      "#ffca28",
                "NO_ACTION":   "#ff5252",
            },
            title="Work Order Decisions",
        )
        st.plotly_chart(fig_d, use_container_width=True)

        # Download
        csv_s = df_out.to_csv(index=False).encode()
        st.download_button("⬇️ Download Sensor Outcomes", csv_s,
                           "sensor_outcomes.csv", "text/csv")


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 5 – Review Queue
# ╚══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔍 Human Review Queue")
    st.caption(
        "Work order requests from Partial Match and LLM Reasoned events require human approval "
        "before being submitted to IFM Hub."
    )

    from pipeline.review_queue import ReviewQueue, ReviewStatus
    rq = ReviewQueue()

    # Seed demo data if queue is empty
    stats = rq.get_stats()
    if stats.get("total", 0) == 0:
        rq.seed_demo_data()
        stats = rq.get_stats()

    # ── Stats ─────────────────────────────────────────────────────────────────
    by_status = stats.get("by_status", {})
    pending_count   = by_status.get("PENDING",   {}).get("count", 0)
    approved_count  = by_status.get("APPROVED",  {}).get("count", 0)
    rejected_count  = by_status.get("REJECTED",  {}).get("count", 0)
    escalated_count = by_status.get("ESCALATED", {}).get("count", 0)

    qs1, qs2, qs3, qs4 = st.columns(4)
    with qs1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">⏳ Pending Review</div><div class="kpi-value partial">{pending_count}</div></div>', unsafe_allow_html=True)
    with qs2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">✅ Approved</div><div class="kpi-value perfect">{approved_count}</div></div>', unsafe_allow_html=True)
    with qs3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">❌ Rejected</div><div class="kpi-value nomatch">{rejected_count}</div></div>', unsafe_allow_html=True)
    with qs4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">🔺 Escalated</div><div class="kpi-value llm">{escalated_count}</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── Filter ────────────────────────────────────────────────────────────────
    status_filter = st.selectbox(
        "Filter by Status",
        ["PENDING", "APPROVED", "REJECTED", "ESCALATED", "ALL"],
        index=0,
    )
    items = rq.get_all(status=None if status_filter == "ALL" else status_filter)

    if not items:
        st.info("No items in queue for the selected status.")
    else:
        st.write(f"**{len(items)} item(s)** found.")
        for item in items:
            match_color = "#ffca28" if item.get("match_type") == "Partial Match" else "#40c4ff"
            with st.expander(
                f"📋 {item.get('asset_name','?')} — {item.get('alert_type','?')} "
                f"| {item.get('match_type','?')} ({item.get('confidence', 0):.1f}%) "
                f"| {item.get('status','?')}"
            ):
                c_left, c_right = st.columns([2, 1])
                with c_left:
                    st.markdown(f"**Asset:** {item.get('asset_name', 'N/A')}")
                    st.markdown(f"**Alert Type:** `{item.get('alert_type', 'N/A')}`")
                    st.markdown(f"**Severity:** {item.get('severity', 'N/A')}")
                    st.markdown(f"**Building:** {item.get('building', 'N/A')}")
                    st.markdown(f"**Match Type:** :orange[{item.get('match_type', 'N/A')}]")
                    st.markdown(f"**Confidence:** {item.get('confidence', 0):.1f}%")
                    st.markdown(f"**Queue ID:** `{item['id']}`")
                    st.markdown(f"**Created:** {item.get('created_at', 'N/A')[:19]}")

                    # Show request payload
                    try:
                        payload = json.loads(item.get("payload_json", "{}"))
                        if payload:
                            st.json(payload)
                    except Exception:
                        pass

                with c_right:
                    if item.get("status") == ReviewStatus.PENDING:
                        reviewer_name = st.text_input(
                            "Reviewer Name", key=f"rev_{item['id']}", value="Operator"
                        )
                        notes_input = st.text_area(
                            "Notes", key=f"notes_{item['id']}", height=80
                        )
                        btn_col1, btn_col2, btn_col3 = st.columns(3)
                        with btn_col1:
                            if st.button("✅ Approve", key=f"app_{item['id']}"):
                                from pipeline.orchestrator import get_pipeline
                                pipe = get_pipeline()
                                pipe.approve_review(item["id"], reviewer=reviewer_name, notes=notes_input)
                                st.success("Approved! Work order sent to IFM Hub.")
                                st.rerun()
                        with btn_col2:
                            if st.button("❌ Reject", key=f"rej_{item['id']}"):
                                rq.reject(item["id"], reviewer=reviewer_name, notes=notes_input)
                                st.warning("Rejected.")
                                st.rerun()
                        with btn_col3:
                            if st.button("🔺 Escalate", key=f"esc_{item['id']}"):
                                rq.escalate(item["id"], reviewer=reviewer_name, notes=notes_input)
                                st.info("Escalated to senior reviewer.")
                                st.rerun()
                    else:
                        st.info(f"Status: **{item.get('status')}**")
                        if item.get("reviewer"):
                            st.markdown(f"Reviewer: {item['reviewer']}")
                        if item.get("review_notes"):
                            st.markdown(f"Notes: _{item['review_notes']}_")

