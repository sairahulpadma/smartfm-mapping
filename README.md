# SmartFM — SFM ↔ IFM AI Mapping Platform

> **Gen AI-powered platform** that maps Smart FM (SFM) assets to IFM Hub assets, classifies sensor alerts into IFM service types, and auto-creates work orders — with a human review gate for ambiguous matches.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-green.svg)](https://github.com/langchain-ai/langgraph)

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup & Installation](#setup--installation)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Using the Dashboard](#using-the-dashboard)
9. [Running Tests & Generating Data](#running-tests--generating-data)
10. [How the AI Pipeline Works](#how-the-ai-pipeline-works)

---

## What It Does

| Feature | Description |
|---|---|
| **Asset Mapping** | Matches SFM assets to IFM Hub assets using a 5-approach fuzzy + LLM cascade |
| **Sensor Monitoring** | Ingests IoT sensor alerts and classifies them into IFM service types |
| **Work Order Creation** | Auto-creates IFM Hub work orders for high-confidence matches |
| **Review Queue** | Queues ambiguous matches for human approval before submitting to IFM Hub |
| **AI Chat** | Natural language Q&A over your mapping results via LangChain |
| **Active Learning** | Approved/rejected reviews feed back into the training dataset |

### 4-Tier Decision Rules

| Tier | Confidence | Action |
|---|---|---|
| 🟢 **Perfect Match** | ≥ 85 % | Work order auto-created in IFM Hub |
| 🟡 **Partial Match** | 50 – 84 % | Queued for human review |
| 🔵 **LLM Reasoned** | AI decides | Queued for human review with AI explanation |
| 🔴 **No Match** | 0 % | Logged only — no request created |

---

## Tech Stack

### AI / ML
| Component | Technology | Purpose |
|---|---|---|
| Primary LLM | **Azure OpenAI GPT-5.5** | Asset matching + service classification |
| Secondary LLM | **Azure Anthropic Claude Sonnet 4.6** | AI chat assistant |
| Agent Orchestration | **LangGraph** | Cascading match pipeline (directed acyclic graph) |
| Chat Agent | **LangChain ReAct** | Natural language Q&A over results |
| Fuzzy Matching | **rapidfuzz** | String similarity (token set ratio) |

### Backend
| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Sensor Event Model | `dataclasses` (typed, serialisable) |
| Review Queue Storage | SQLite (demo) / Azure Service Bus (production) |
| HTTP Client | `httpx` |
| Config / Secrets | `python-dotenv` |

### Frontend
| Component | Technology |
|---|---|
| Dashboard | **Streamlit** (5-tab UI) |
| Charts | **Plotly Express** |

### Data
| Component | Technology |
|---|---|
| Training dataset | 1 200-row CSV/JSONL (sensor → service classification) |
| Asset data | Excel files via `openpyxl` |
| Service catalog | JSON (20 IFM service classifications) |

---

## Project Structure

```
smartfm-mapping/
├── app.py                          # Streamlit dashboard (5 tabs)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
├── ARCHITECTURE.md                 # HLD / LLD / flow diagrams
│
├── pipeline/
│   ├── data_loader.py              # Load SFM/IFM Excel files
│   ├── matcher.py                  # 5-approach fuzzy scoring engine
│   ├── langgraph_agent.py          # LangGraph cascading matcher
│   ├── sensor_ingestor.py          # IoT sensor event parser & enricher
│   ├── service_classifier.py       # 4-tier service classification engine
│   ├── work_order_creator.py       # IFM Hub API client + decision engine
│   ├── review_queue.py             # SQLite human review queue
│   └── orchestrator.py             # End-to-end pipeline orchestration
│
├── llm/
│   ├── chat_agent.py               # LangChain ReAct chat agent
│   ├── service_classification_agent.py  # LLM service classification fallback
│   └── prompts.py                  # System prompts & few-shot examples
│
├── data/
│   ├── service_catalog.json        # 20 IFM service classifications
│   ├── pipeline_outcomes.jsonl     # Audit log (auto-generated)
│   ├── review_queue.db             # SQLite queue (auto-generated)
│   ├── test/
│   │   ├── demo_results.csv        # Pre-computed demo data (all 4 tiers)
│   │   ├── test_sfm_demo.xlsx      # 28 SFM test records
│   │   ├── test_ifm_demo.xlsx      # 33 IFM test records
│   │   ├── make_demo_csv.py        # Regenerate demo_results.csv
│   │   └── generate_test_data.py   # Regenerate test Excel files
│   └── training/
│       ├── sensor_training_data.csv     # 1 200-row labeled training data
│       ├── sensor_training_data.jsonl   # JSONL for LLM fine-tuning
│       ├── sensor_training_data.json    # First 150 rows as few-shot examples
│       ├── training_examples.json       # Asset-mapping few-shot examples
│       ├── active_learning.jsonl        # Operator review labels (grows over time)
│       └── generate_sensor_training.py  # Regenerate training data
│
├── gpt5.5 sample.py                # Azure OpenAI GPT-5.5 quickstart
└── sonnet4.6 sample.py             # Azure Anthropic Claude 4.6 quickstart
```

---

## Prerequisites

- **Python 3.9 or higher**
- **pip** (or `pip3`)
- Azure OpenAI access (GPT-5.5 deployment) — for LLM matching features
- Azure Anthropic access (Claude Sonnet 4.6) — for the AI chat assistant
- *(Optional)* IFM Hub API credentials — for real work order creation (demo mode works without it)

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/sairahulpadma/smartfm-mapping.git
cd smartfm-mapping
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On macOS with Python 3.9 (system), you may need `pip3 install -r requirements.txt`

### 4. Generate training & test data

```bash
# Generate 1 200-row sensor training dataset
python3 data/training/generate_sensor_training.py

# Generate test Excel files (28 SFM + 33 IFM records)
python3 data/test/generate_test_data.py

# Generate pre-computed demo results (all 4 match tiers)
python3 data/test/make_demo_csv.py
```

---

## Configuration

### 1. Copy the example environment file

```bash
cp .env.example .env
```

### 2. Fill in your credentials in `.env`

```ini
# Azure OpenAI (GPT-5.5)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.5_1
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Anthropic (Claude Sonnet 4.6)
AZURE_ANTHROPIC_ENDPOINT=https://<your-resource>.services.ai.azure.com/anthropic/
AZURE_ANTHROPIC_API_KEY=<your-key>

# IFM Hub (leave as-is for demo mode)
IFM_BASE_URL=https://api.ifm-hub.example.com
IFM_API_KEY=<your-key>
```

> **The app runs fully in demo mode without any API keys** — just skip filling in the keys and use the "Load Demo Results" and "Run Demo Sensor Events" buttons.

---

## Running the Application

```bash
streamlit run app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

If `streamlit` is not on your PATH (common with Python 3.9 on macOS):

```bash
/Users/$USER/Library/Python/3.9/bin/streamlit run app.py --server.port 8501
```

---

## Using the Dashboard

The dashboard has **5 tabs**:

### Tab 1 — Upload & Map
- Upload separate SFM + IFM Excel files, or a combined hackathon Excel
- **OR** click **"🎯 Load Demo Results"** to instantly load pre-computed results covering all 4 match tiers — no upload needed
- Click **"🚀 Run AI Mapping Pipeline"** to process uploaded files
- Filter results by match type, confidence, and asset name
- Download results as CSV

### Tab 2 — Dashboard
- KPI cards: Total / Perfect / Partial / LLM Reasoned / No Match counts
- Donut chart: match type distribution
- Histogram: confidence score distribution
- Stacked bar: match quality per building
- Horizontal bar: top 10 lowest-confidence assets

### Tab 3 — AI Assistant
- Ask natural language questions about your mapping results
- Example prompts: *"Which assets have no match?"*, *"Show me low confidence assets"*, *"Give me a summary"*
- Powered by LangChain ReAct + Claude Sonnet 4.6 (falls back to keyword answers if LLM unavailable)

### Tab 4 — Sensor Monitor
- Click **"🚀 Run Demo Sensor Events"** to simulate 5 IoT sensor alerts through the full pipeline
- Use **"➕ Simulate a Custom Sensor Alert"** to submit any asset type + alert type
- See real-time decisions: Auto-Created ✅ / Pending Review ⚠️ / No Action ❌

### Tab 5 — Review Queue
- Lists all Partial Match and LLM Reasoned events awaiting human review
- **Approve** → work order submitted to IFM Hub + logged to active learning dataset
- **Reject** → logged, no request created
- **Escalate** → flagged for senior reviewer

---

## Running Tests & Generating Data

```bash
# Re-generate 1 200 training rows
python3 data/training/generate_sensor_training.py

# Re-generate test Excel files
python3 data/test/generate_test_data.py

# Re-generate demo CSV (all 4 tiers guaranteed)
python3 data/test/make_demo_csv.py

# Quick pipeline smoke test
python3 -c "
from pipeline.sensor_ingestor import make_demo_events
from pipeline.orchestrator import SensorToWorkOrderPipeline
pipe = SensorToWorkOrderPipeline(use_llm=False, demo_mode=True)
outcomes = pipe.process_batch(make_demo_events())
for o in outcomes:
    print(o['asset_name'], '|', o['match_type'], '|', o['decision'])
print('Summary:', pipe.get_summary())
"
```

---

## How the AI Pipeline Works

### Asset Mapping (SFM ↔ IFM)

Uses a **LangGraph** DAG with 5 cascading approaches:

```
Approach 1: Name + Make/Model/Serial + Location         → Perfect Match
Approach 2: Name + Location + Building                  → Perfect Match
Approach 3: Make/Model/Serial + Location + Building     → Perfect Match
Approach 4: Name (≥50%) + Location + Building           → Partial Match
Approach 5: Name (≥50%) + Make/Model + Location + Bldg → Partial Match
LLM Node:   GPT-5.5 JSON-mode reasoning                 → LLM Reasoned
```

Each node short-circuits to END if a match is found, otherwise passes through.

### Sensor → Work Order

```
IoT Sensor Alert
  ↓
SensorIngestor  — parse, enrich, validate
  ↓
ServiceClassifier (4-tier cascade)
  ├─ Exact catalog match (≥85%)   → Perfect Match
  ├─ Fuzzy keyword match (50-84%) → Partial Match
  ├─ LLM agent (<50%)             → LLM Reasoned
  └─ No match                     → No Match
  ↓
WorkOrderCreator / ReviewQueue
  ├─ Perfect → POST /requests to IFM Hub (auto)
  ├─ Partial/LLM → SQLite review queue → human review
  └─ No Match → log only
```

---

## Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`
- All API keys must be stored in `.env` only
- The review queue uses parameterised SQL queries (no injection risk)
- All auto-created work orders are logged to `data/pipeline_outcomes.jsonl` for audit

---

*Built with Azure OpenAI GPT-5.5 · Claude Sonnet 4.6 · LangGraph · LangChain · Streamlit*
