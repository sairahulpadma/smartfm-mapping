# SFM ↔ IFM AI Platform — Copilot Instructions

> This file tells GitHub Copilot exactly how this codebase is structured,
> what conventions to follow, and how to implement new features correctly.
> Edit the sections marked **[EXTEND HERE]** to add new feature requests.

---

## 1. What This Platform Does

A Gen-AI engineering platform with two connected pipelines:

| Pipeline | What it does |
|---|---|
| **Asset Mapping** | Match SFM assets to IFM Hub assets using a 5-approach fuzzy + LLM cascade (LangGraph) |
| **Sensor → Work Order** | Ingest IoT sensor alerts → classify to IFM service → auto-create work order or queue for human review |

---

## 2. Tech Stack (never change these without asking)

| Layer | Technology |
|---|---|
| Dashboard | Streamlit ≥ 1.32, Plotly, pandas |
| AI / LLM | Azure OpenAI GPT-5.5 (`gpt-5.5_1`), Azure Anthropic Claude Sonnet 4.6 |
| LLM Framework | LangChain ReAct agent, LangGraph StateGraph |
| Fuzzy matching | rapidfuzz `fuzz.token_set_ratio` |
| Pipeline | Python dataclasses, pure Python orchestration |
| Storage | SQLite (`data/review_queue.db`) for review queue, JSONL for pipeline outcomes & active learning |
| HTTP client | httpx (async-capable) for IFM Hub API calls |
| Config | python-dotenv, all secrets via `os.getenv()` — **never hardcode keys** |
| Python | 3.9, macOS |

---

## 3. Folder Structure & What Each File Does

```
smartfm-mapping/
│
├── app.py                          ← Streamlit dashboard (5 tabs)
├── requirements.txt                ← All pip dependencies
├── .env.example                    ← Template — copy to .env and fill keys
│
├── pipeline/
│   ├── __init__.py
│   ├── langgraph_agent.py          ← 7-node LangGraph asset-matching graph
│   ├── matcher.py                  ← 5 fuzzy scoring functions (approach1–partial2)
│   ├── sensor_ingestor.py          ← SensorEvent dataclass + SensorIngestor class
│   ├── service_classifier.py       ← 4-tier classification engine (Perfect/Partial/LLM/NoMatch)
│   ├── work_order_creator.py       ← IFM Hub API client + decide_action() logic
│   ├── review_queue.py             ← SQLite human review queue
│   ├── orchestrator.py             ← Wires all pipeline stages end-to-end
│   └── data_loader.py              ← Reads SFM/IFM Excel uploads
│
├── llm/
│   ├── __init__.py
│   ├── chat_agent.py               ← LangChain ReAct chat agent (Tab 3)
│   ├── prompts.py                  ← All LLM system prompts
│   └── service_classification_agent.py  ← LLM fallback for Tier 3 classification
│
└── data/
    ├── service_catalog.json        ← 20 IFM service classifications
    ├── pipeline_outcomes.jsonl     ← Append-only log of every pipeline run
    ├── review_queue.db             ← SQLite review queue (auto-created)
    ├── test/
    │   ├── demo_results.csv        ← Pre-built demo output (28 rows, all 4 tiers)
    │   ├── generate_test_data.py   ← Generates test_sfm_demo.xlsx + test_ifm_demo.xlsx
    │   └── make_demo_csv.py        ← Hardcodes demo_results.csv (no pipeline run needed)
    └── training/
        ├── sensor_training_data.csv/.jsonl/.json  ← 1200-row labeled dataset
        ├── generate_sensor_training.py            ← Generates the training data
        ├── training_examples.json                 ← Asset mapping training examples
        └── active_learning.jsonl                  ← Approved decisions fed back to model
```

---

## 4. Streamlit Dashboard — 5 Tabs

| Tab | Name | What it shows |
|---|---|---|
| 1 | Upload & Map | File upload → LangGraph pipeline → results table with colour-coded tiers |
| 2 | Dashboard | KPI cards, Plotly pie chart (match type distribution), bar chart (building analysis) |
| 3 | AI Chat | LangChain ReAct agent answering natural language questions about the data |
| 4 | Sensor Monitor | Demo sensor events, custom alert form, decision outcome KPIs, pie chart |
| 5 | Review Queue | SQLite queue — Approve / Reject / Escalate buttons per pending item |

### Conventions for adding a new tab
1. Add to `st.tabs([...])` list in `app.py`
2. All tab content goes inside `with tabN:` block
3. Use the existing `.kpi-card` CSS class for metric boxes
4. Session state keys follow pattern: `st.session_state["snake_case_name"]`
5. For Plotly: always do `df.value_counts().reset_index(); df.columns = ["Label", "Count"]` to avoid pandas version issues with column naming

---

## 5. Pipeline Architecture — The Cascade

### Asset Mapping (LangGraph — `pipeline/langgraph_agent.py`)

```
START → approach1 → approach2 → approach3 → partial1 → partial2 → llm_reason → finalize → END
         ↓ stop early if match_found = True at any node
```

- Each node calls a scorer from `matcher.py`
- Scorers use `rapidfuzz.fuzz.token_set_ratio`
- Thresholds: Perfect needs name ≥ 90, Partial needs name ≥ 50
- LLM node sends top-5 candidates to GPT-5.5 and asks for a match decision
- State object: `AssetMappingState` TypedDict

### Sensor → Work Order (orchestrator — `pipeline/orchestrator.py`)

```
SensorEvent → ServiceClassifier (4 tiers) → decide_action() → AUTO_CREATE / REVIEW / NO_ACTION
                                                    ↓
                                        WorkOrderCreator (IFM Hub API)
                                        ReviewQueue (SQLite)
```

**4 classification tiers:**
| Tier | Confidence | Action |
|---|---|---|
| Perfect ≥ 85% | High composite score | AUTO_CREATE work order |
| Partial 50–84% | Medium score | REVIEW queue |
| LLM 30–49% | LLM fallback call | REVIEW queue |
| No Match < 30% | Nothing matches | NO_ACTION, log only |

---

## 6. Key Data Structures

### SensorEvent (sensor_ingestor.py)
```python
@dataclass
class SensorEvent:
    event_id: str
    asset_id: str
    asset_name: str
    asset_type: str          # e.g. "AHU", "Electrical Panel"
    alert_type: str          # e.g. "temp_high", "breaker_trip"
    severity: AlertSeverity  # CRITICAL / HIGH / MEDIUM / LOW
    description: str
    location: str
    building: str
    timestamp: str
    sensor_value: float
    unit: str
    threshold_value: float
    metadata: dict
```

### ServiceCatalog entry (data/service_catalog.json)
```json
{
  "id": "HVAC-COOL-001",
  "name": "HVAC Cooling System Maintenance",
  "category": "HVAC",
  "subcategory": "Cooling",
  "asset_types": ["AHU", "Chiller", "Cooling Tower"],
  "alert_types": ["temp_high", "temp_deviation"],
  "keywords": ["cooling", "chiller", "refrigerant"],
  "priority": "HIGH",
  "sla_hours": 4,
  "auto_create_threshold": 85
}
```

### IFM Hub Work Order Request Payload (work_order_creator.py)
```json
{
  "requestTypeId": "...",
  "summary": "...",
  "description": "...",
  "priority": "HIGH",
  "assetId": "...",
  "locationId": "...",
  "requestorId": "...",
  "orgIds": [...],
  "metadata": { "source": "sensor_pipeline", "confidence": 92.5 }
}
```

---

## 7. Environment Variables (all in .env)

```
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://....cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5.5_1
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_ANTHROPIC_API_KEY=
AZURE_ANTHROPIC_ENDPOINT=https://....services.ai.azure.com/anthropic/
AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-4-6

IFM_BASE_URL=https://your-ifm-instance.com/api
IFM_API_KEY=
IFM_TENANT_ID=
IFM_ORG_IDS=["org-001"]
IFM_REQUESTOR_ID=
PERFECT_THRESHOLD=85
```

---

## 8. Coding Conventions

- **No hardcoded secrets** — always `os.getenv("VAR_NAME")`, raise `EnvironmentError` if required
- **Dataclasses for data models** — use `@dataclass` not plain dicts for pipeline objects
- **Parameterised SQL** — always `cursor.execute("... WHERE id = ?", (id,))` — never f-strings in SQL
- **Graceful fallback** — every LLM call must have a `try/except` that returns a safe default
- **Append-only logs** — write outcomes to `.jsonl` files with `mode="a"`
- **No global mutable state** except for the module-level pipeline singleton in `orchestrator.py`
- **Plotly column naming** — always assign `.columns = ["Label", "Count"]` after `value_counts().reset_index()` to avoid pandas version issues
- **Session state** — initialise all keys in the `# Session state init` block at the top of `app.py`
- **Tab content** — keep all tab logic inside `with tabN:` blocks, no tab-specific functions at module level

---

## 9. Adding New Features — Checklist

When asked to add a new feature, follow this order:

1. **Pipeline change?** → edit the right file in `pipeline/` or `llm/`
2. **New data model?** → add `@dataclass` in the relevant pipeline file
3. **New service type?** → add entry to `data/service_catalog.json`
4. **New tab?** → add to `st.tabs()` and add `with tabN:` block in `app.py`
5. **New chart?** → use Plotly Express, follow the column-naming convention above
6. **New env var?** → add to `.env.example` with a comment explaining it
7. **Training data?** → update `data/training/generate_sensor_training.py` and re-run it
8. **Demo CSV?** → update `data/test/make_demo_csv.py` rows list to include new match types
9. **Architecture doc?** → update `ARCHITECTURE.md` Section 2 (HLD) or Section 3 (LLD) Mermaid diagrams

---

## 10. Running the Platform

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate test & demo data
python3 data/training/generate_sensor_training.py
python3 data/test/generate_test_data.py
python3 data/test/make_demo_csv.py

# 3. Set up environment
cp .env.example .env
# Edit .env and fill in your Azure keys

# 4. Launch dashboard
streamlit run app.py --server.port 8501
# Open http://localhost:8501
```

---

## 11. [EXTEND HERE] — New Feature Requests

> Add your next feature instructions below this line.
> Describe: what it should do, which files it touches, and any new data/API requirements.

<!--
EXAMPLE FORMAT:

### Feature: Real-time MQTT Sensor Feed
- Replace demo event buttons in Tab 4 with a live MQTT subscriber
- New file: pipeline/mqtt_listener.py
- Uses: paho-mqtt library
- Env vars needed: MQTT_BROKER_URL, MQTT_TOPIC, MQTT_USERNAME, MQTT_PASSWORD
- On new message: call orchestrator.process_raw_event() and refresh Tab 4
-->
