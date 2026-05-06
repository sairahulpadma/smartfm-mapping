# SFM ↔ IFM AI Platform — Architecture Document

> Version 2.0 | May 2026 | Hackathon Edition

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Design (HLD)](#2-high-level-design-hld)
3. [Low-Level Design (LLD)](#3-low-level-design-lld)
4. [End-to-End Flow Diagrams](#4-end-to-end-flow-diagrams)
5. [Component Reference](#5-component-reference)
6. [Tech Stack](#6-tech-stack)
7. [Data Models](#7-data-models)
8. [IFM Hub API Contract](#8-ifm-hub-api-contract)
9. [Training Pipeline & Active Learning](#9-training-pipeline--active-learning)
10. [Security & Production Considerations](#10-security--production-considerations)
11. [Deployment Guide](#11-deployment-guide)

---

## 1. Executive Summary

The platform solves two connected problems:

| Problem | Solution |
|---|---|
| **Asset Mapping** — SFM assets need to be matched to IFM Hub assets | 8-node LangGraph fuzzy + LLM cascade (approach1 → llm_reason) |
| **Sensor → Work Order** — When a sensor alerts, the right IFM service must be found and a work order auto-created or queued | 4-tier classification engine + WorkOrderCreator |

### Decision Rules (applies to both pipelines)

| Tier | Confidence | Action |
|---|---|---|
| 🟢 **Perfect Match** | ≥ 85 % | Auto-create work order in IFM Hub |
| 🟡 **Partial + LLM Verified** | 50 – 84 % + GPT-5.5 confirms | Auto-create (boosted) or queue for review |
| 🔵 **LLM Reasoned** | Claude Sonnet 4.6 decides | Queue for human review with AI explanation |
| 🔴 **No Match** | 0 % | Log only — no request created |

### Model Routing Strategy

| Node / Purpose | Model | Rationale |
|---|---|---|
| `llm_verify_partial` (asset matching) | **GPT-5.5** | Short JSON confirm/reject — fast |
| `llm_reason` (asset matching) | **Claude Sonnet 4.6** | Nuanced ambiguity reasoning |
| Tier 2 LLM verify (service classification) | **GPT-5.5** | Short JSON confirm/reject |
| Tier 3 LLM classify (service classification) | **GPT-5.5** | Structured JSON output |
| Tab 3 chat agent | **Claude Sonnet 4.6** | Multi-turn Q&A, long context |

---

## 2. High-Level Design (HLD)

```mermaid
graph TB
    subgraph SENSORS["🌡️ Sensor Layer"]
        S1[IoT Sensor — HVAC]
        S2[IoT Sensor — Electrical]
        S3[IoT Sensor — Plumbing]
        S4[IoT Sensor — Other]
    end

    subgraph INGESTION["📡 Ingestion Layer"]
        IH["Demo Events / Webhook"]
        SI[SensorIngestor\nsensor_ingestor.py\n35-event demo factory]
    end

    subgraph CLASSIFICATION["🧠 Classification Engine"]
        SC[ServiceClassifier\n4-tier cascade\nservice_classifier.py]
        LLM_SC[LLM Classification Agent\nservice_classification_agent.py\nGPT-5.5 Tier 3]
        CAT[(Service Catalog\nservice_catalog.json\n23 classifications)]
    end

    subgraph DECISION["⚡ Decision Engine"]
        WOC[WorkOrderCreator\nwork_order_creator.py]
        RQ[(Review Queue\nSQLite review_queue.db)]
    end

    subgraph IFM["🏢 IFM Hub"]
        API[IFM Hub REST API\nPOST /requests]
        WO[(Work Orders DB)]
    end

    subgraph ASSET_MAP["🗺️ Asset Mapping"]
        MATCH[LangGraph 8-node Pipeline\nlanggraph_agent.py]
        SFM_DB[(SFM Assets\nExcel upload)]
        IFM_DB[(IFM Assets\nExcel upload)]
    end

    subgraph LLM_LAYER["🤖 LLM Layer"]
        GPT[Azure OpenAI GPT-5.5\nverify + classify]
        CLAUDE[Azure Anthropic\nClaude Sonnet 4.6\nreason + chat]
        METRICS[LLM Metrics Tracker\nllm_metrics.jsonl]
    end

    subgraph DASHBOARD["📊 Streamlit Dashboard (6 tabs)"]
        T1[Tab 1: Upload & Map]
        T2[Tab 2: Dashboard]
        T3[Tab 3: AI Assistant]
        T4[Tab 4: Sensor Monitor]
        T5[Tab 5: Review Queue]
        T6[Tab 6: LLM Analytics]
    end

    S1 & S2 & S3 & S4 --> IH
    IH --> SI
    SI --> SC
    SC -- "Tier 2/3" --> LLM_SC
    CAT --> SC
    LLM_SC --> GPT
    SC -- "Perfect ≥85%" --> WOC
    SC -- "Partial/LLM" --> RQ
    SC -- "No Match" --> DASHBOARD
    WOC --> API
    API --> WO
    RQ --> T5
    WO --> T4
    MATCH --> T1 & T2
    SFM_DB & IFM_DB --> MATCH
    MATCH --> CLAUDE
    MATCH --> GPT
    T3 --> CLAUDE
    GPT & CLAUDE --> METRICS
    METRICS --> T6
```

---

## 3. Low-Level Design (LLD)

### 3.1 Asset Mapping — 8-Node LangGraph Pipeline

```mermaid
flowchart LR
    START([START]) --> A1[approach1\nName+Make/Model\n+Serial+Location]
    A1 -->|match_found=True| FIN
    A1 --> A2[approach2\nName+Location\n+Building]
    A2 -->|match_found=True| FIN
    A2 --> A3[approach3\nMake/Model/Serial\n+Location+Building]
    A3 -->|match_found=True| FIN
    A3 --> P1[partial1\nName≥50%\n+Location+Building]
    P1 --> P2[partial2\nName≥50%+Make/Model\n+Location+Building]
    P2 --> LVP[llm_verify_partial\nGPT-5.5 JSON\nconfirm/reject]
    LVP -->|confirmed| FIN
    LVP -->|rejected| LLM[llm_reason\nClaude Sonnet 4.6\ntop-3 candidates]
    P1 -->|no match| P2
    LLM --> FIN[finalize\nNo Match if none]
    FIN --> END([END])
```

**Thresholds:**
- Perfect match: name similarity ≥ 90 (token_set_ratio)
- Partial match: name similarity ≥ 50
- `llm_verify_partial` runs after any partial1/partial2 hit — GPT-5.5 confirms or rejects
- `llm_reason` receives top-3 candidates (compact JSON, no indent) — Claude makes final call

**API constraints (GPT-5.5):**
- Parameter is `max_completion_tokens` (not `max_tokens`)
- `temperature` must be omitted (only supports default=1)

**Claude auth:**
- Must use `AnthropicFoundry` class (not `anthropic.Anthropic`) — Azure AI Foundry uses different auth headers

### 3.2 Service Classification Engine — 4-Tier Cascade

```mermaid
flowchart TD
    A([Sensor Event]) --> B[Score all 23 service classifications\nweighted: asset_type 45% + alert_type 40% + keywords 15%]

    B --> D{Best score?}
    D -- "≥ 85%\nTier 1 Perfect" --> E[✅ Return Perfect Match\nauto_create_threshold check]
    D -- "50–84%\nTier 2 Partial" --> F[🔍 GPT-5.5 LLM Verify\nconfirm or reject]
    F -- confirmed --> G[✅ Partial + LLM Verified]
    F -- rejected --> H[🤖 Tier 3 LLM Classify]
    D -- "30–49%\nTier 3 Ambiguous" --> H
    H --> I{LLM chooses valid ID?}
    I -- Yes --> J[🔵 LLM Reasoned]
    I -- No  --> K[❌ No Match]
    D -- "< 30%\nTier 4" --> K
```

### 3.3 Work Order Decision Flow

```mermaid
flowchart TD
    A([Classification Result]) --> B{match_type?}

    B -- "Perfect / Partial+LLM\nconfidence ≥ auto_create_threshold" --> C[Build IFM Hub payload]
    C --> D[POST /requests to IFM Hub API]
    D --> E[(pipeline_outcomes.jsonl)]
    D --> F[✅ AUTO_CREATE\nrequest_id returned]

    B -- "Partial or LLM Reasoned" --> G[Build request payload]
    G --> H[(Review Queue SQLite)]
    H --> I[⚠️ REVIEW\nOperator sees in Tab 5]
    I --> J{Decision}
    J -- Approve --> K[POST /requests to IFM Hub]
    J -- Reject  --> L[❌ Logged only]
    J -- Escalate --> M[🔺 Senior reviewer]
    K --> N[(active_learning.jsonl)]

    B -- "No Match" --> O[❌ NO_ACTION\nLog only]
```

---

## 4. End-to-End Flow Diagrams

### 4.1 Happy Path — Perfect Match Auto Work Order

```mermaid
sequenceDiagram
    participant Sensor as 🌡️ IoT Sensor
    participant Ingestor as SensorIngestor
    participant Classifier as ServiceClassifier
    participant GPT as GPT-5.5
    participant Creator as WorkOrderCreator
    participant IFM as IFM Hub API

    Sensor->>Ingestor: {asset_type:"AHU", alert:"temperature_high", reading:78.5°F}
    Ingestor->>Ingestor: parse() → SensorEvent dataclass
    Ingestor->>Classifier: classify(event)
    Classifier->>Classifier: score 23 catalog entries
    Classifier-->>Creator: {match_type:"Perfect", confidence:94%, sc:"HVAC - Cooling System Failure"}
    Creator->>Creator: decide_action() → AUTO_CREATE
    Creator->>IFM: POST /requests {serviceClassificationId, description, locationId, ...}
    IFM-->>Creator: {status:"created", request_id:"req-abc"}
```

### 4.2 Partial Match — LLM Verify → Review

```mermaid
sequenceDiagram
    participant Classifier as ServiceClassifier
    participant GPT as GPT-5.5
    participant Queue as ReviewQueue
    participant Op as 👤 Operator

    Classifier->>GPT: {asset="Chiller", alert="low_flow_fault", fuzzy_match="HVAC Cooling 62%"}
    GPT-->>Classifier: {confirmed:true, adjusted_confidence:78, reasoning:"..."}
    Classifier->>Queue: INSERT PENDING {match_type:"Partial + LLM Verified", confidence:78%}
    Queue-->>Op: Shows in Tab 5 Review Queue
    Op->>Queue: approve(reviewer="J.Smith")
    Queue->>Queue: UPDATE APPROVED + append active_learning.jsonl
```

### 4.3 LLM Reasoning Path (Asset Mapping)

```mermaid
sequenceDiagram
    participant Graph as LangGraph
    participant P1 as partial1/partial2
    participant GPT as GPT-5.5
    participant Claude as Claude Sonnet 4.6

    Graph->>P1: score(sfm_asset, ifm_candidates)
    P1-->>Graph: name_similarity=65% (Partial hit)
    Graph->>GPT: llm_verify_partial {fuzzy_match, score=65%}
    GPT-->>Graph: {confirmed:false, reasoning:"different building"}
    Graph->>Claude: llm_reason {sfm_asset, top_3_candidates}
    Claude-->>Graph: {match:true, asset_id:"IFM-XXX", confidence:71, reasoning:"..."}
    Graph-->>Graph: match_type="LLM Reasoned"
```

---

## 5. Component Reference

| File | Purpose | Notes |
|---|---|---|
| `app.py` | Streamlit dashboard — 6 tabs | Hot-reloads on save |
| `pipeline/langgraph_agent.py` | 8-node LangGraph asset matcher | Claude for llm_reason, GPT for llm_verify_partial |
| `pipeline/matcher.py` | 5 fuzzy scoring functions | `fuzz.token_set_ratio` |
| `pipeline/sensor_ingestor.py` | SensorEvent dataclass + 35-event demo factory | All 4 tiers + multi-asset groups |
| `pipeline/service_classifier.py` | 4-tier service classification engine | Tier 2 LLM verify via GPT-5.5 |
| `pipeline/work_order_creator.py` | IFM Hub API client + `decide_action()` | Demo mode mock included |
| `pipeline/review_queue.py` | SQLite human review queue | Parameterised SQL only |
| `pipeline/orchestrator.py` | End-to-end pipeline + `process_batch()` | 300ms inter-event delay for rate limit protection |
| `llm/chat_agent.py` | LangChain ReAct chat agent | Claude primary, GPT fallback; `anthropic_api_key` + `anthropic_api_url` params |
| `llm/metrics_tracker.py` | Singleton LLM call tracker | Thread-safe, append-only JSONL |
| `llm/service_classification_agent.py` | GPT-5.5 Tier 3 LLM classify | Compact prompt, `max_completion_tokens=256` |
| `llm/prompts.py` | System prompts & few-shot examples | |
| `data/service_catalog.json` | 23 IFM service classifications | Includes 3 multi-asset entries |
| `data/llm_metrics.jsonl` | LLM call audit log | Append-only, read by Tab 6 |
| `data/pipeline_outcomes.jsonl` | Pipeline run audit log | Append-only |
| `data/test/generate_test_data.py` | Generates 35 SFM + 39 IFM test records | All 8 match type variants |
| `data/test/make_demo_csv.py` | Hardcodes demo_results.csv | No pipeline run needed |
| `data/training/generate_sensor_training.py` | Generates 1,200-row labeled training data | |

---

## 6. Tech Stack

### Core AI/ML
| Component | Technology | Purpose |
|---|---|---|
| LLM (GPT) | **Azure OpenAI GPT-5.5** (`gpt-5.5_1`) | LLM verify + service classification |
| LLM (Claude) | **Azure Anthropic Claude Sonnet 4.6** | Asset matching reasoning + chat |
| Claude SDK | `AnthropicFoundry` (from `anthropic` package) | Azure AI Foundry auth — NOT `anthropic.Anthropic` |
| Agent Orchestration | **LangGraph StateGraph** | 8-node cascading pipeline |
| Chat Agent | **LangChain ReAct** | NL Q&A over results |
| Fuzzy Matching | **rapidfuzz** `token_set_ratio` | String similarity |

### Backend
| Component | Technology | Purpose |
|---|---|---|
| Language | **Python 3.9+** | All pipeline code |
| Event Model | **dataclasses** | Typed `SensorEvent` |
| Queue | **SQLite** | Review queue (demo) |
| HTTP | **httpx** | IFM Hub API calls |
| Config | **python-dotenv** | Secrets from `.env` |

### Frontend
| Component | Technology | Purpose |
|---|---|---|
| Dashboard | **Streamlit ≥ 1.32** | 6-tab interactive UI |
| Charts | **Plotly Express** | All visualisations |

---

## 7. Data Models

### SensorEvent (pipeline/sensor_ingestor.py)
```python
@dataclass
class SensorEvent:
    event_id:    str            # UUID
    sensor_id:   str            # Physical sensor ID
    asset_id:    str            # SFM asset reference
    asset_name:  str            # Human-readable name
    asset_type:  str            # AHU / Chiller / Pump / BMS / etc.
    alert_type:  str            # temperature_high / power_failure / zone_failure / etc.
    severity:    AlertSeverity  # CRITICAL / HIGH / MEDIUM / LOW
    reading:     SensorReading  # value, unit, threshold_min, threshold_max
    location_id: str            # IFM Hub location UUID
    building:    str
    floor:       str
    room:        str
    timestamp:   str            # ISO 8601
    metadata:    dict
```

### AssetMappingState (pipeline/langgraph_agent.py)
```python
class AssetMappingState(TypedDict):
    sfm_record:       dict
    ifm_records:      List[dict]
    match_found:      bool
    match_result:     Optional[dict]
    match_type:       str
    confidence:       float
    reasoning:        str
    llm_verify_note:  str          # GPT-5.5 verify_partial note
    approaches_tried: List[str]
```

### ServiceCatalog entry (data/service_catalog.json)
```json
{
  "id": "4aa86b28-c506-11ed-afa1-0242ac120002",
  "name": "HVAC - Cooling System Failure",
  "category": "HVAC",
  "subcategory": "Cooling",
  "asset_types": ["AHU", "FCU", "Chiller", "AC Unit", "CRAC", "HVAC", "Air Handler", "Fan Coil"],
  "alert_types": ["temperature_high", "cooling_failure", "refrigerant_leak", "compressor_fault", "condenser_fault"],
  "keywords": ["cooling", "chiller", "refrigerant", "compressor"],
  "priority": "HIGH",
  "sla_hours": 4,
  "auto_create_threshold": 85
}
```

### IFM Hub Request Payload (work_order_creator.py)
```json
{
  "orgs": ["uuid1"],
  "tenantId": "601205a7f110dd542d9237bc",
  "description": "Auto-generated: AHU-1 | temperature_high | 78.5°F (threshold: 72°F)",
  "locationId": "f754334d-17cc-4890-bc58-2a4e1a386549",
  "serviceClassificationId": "4aa86b28-c506-11ed-afa1-0242ac120002",
  "requestorId": "cb0795b9-32d6-4574-8a88-bd8fc5b1b5cd",
  "source": "AI Sensor Alert",
  "sourceApp": "sfm-ai-platform",
  "_meta": {
    "sensor_event_id": "uuid",
    "match_type": "Perfect Match",
    "match_confidence": 92.0,
    "ai_reasoning": "..."
  }
}
```

---

## 8. IFM Hub API Contract

```
POST {IFM_BASE_URL}/requests
Authorization: Bearer {IFM_API_KEY}
Content-Type: application/json
Body: <IFM Hub Request Payload above>
```

### Environment Variables (`.env`)
```ini
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.5_1
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Anthropic (Claude) — Azure AI Foundry endpoint
AZURE_ANTHROPIC_ENDPOINT=https://<resource>.services.ai.azure.com/anthropic/
AZURE_ANTHROPIC_API_KEY=<key>
AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-4-6

# IFM Hub
IFM_BASE_URL=https://api.ifm-hub.example.com
IFM_API_KEY=<key>
IFM_TENANT_ID=601205a7f110dd542d9237bc
IFM_ORG_IDS=["7bb1b889-..."]
IFM_REQUESTOR_ID=cb0795b9-32d6-4574-8a88-bd8fc5b1b5cd

# Pipeline thresholds
PERFECT_THRESHOLD=85
```

---

## 9. Training Pipeline & Active Learning

```
data/training/
├── training_examples.json          ← 13 asset-mapping few-shot examples (SFM↔IFM)
├── sensor_training_data.csv        ← 1,200 sensor classification rows
├── sensor_training_data.jsonl      ← Same, JSONL for LLM fine-tuning
├── sensor_training_data.json       ← First 150 rows as few-shot examples
└── active_learning.jsonl           ← Operator approvals/rejections (grows over time)
```

### Active Learning Loop

```mermaid
flowchart LR
    A[Sensor Event] --> B[Classification]
    B --> C{Partial/LLM?}
    C -- Yes --> D[Review Queue]
    D --> E[Operator Reviews]
    E -- Approve --> F[Work Order Created]
    E -- Reject  --> G[Discarded]
    F & G --> H[(active_learning.jsonl\nlabel=approved/rejected)]
    H --> I[Periodic prompt improvement\nor fine-tuning]
    I --> B
```

### Training Data Distribution (1,200 rows)
| Match Type | Count | % |
|---|---|---|
| Perfect Match | 569 | 47.4% |
| LLM Reasoned | 232 | 19.3% |
| Partial Match | 210 | 17.5% |
| No Match | 189 | 15.8% |

---

## 10. Security & Production Considerations

| Area | Concern | Mitigation |
|---|---|---|
| **API Keys** | Hardcoded credentials | All keys in `.env` / Azure Key Vault in prod |
| **Input Validation** | Malicious sensor payloads | Schema validation on all SensorEvent fields |
| **IFM API Auth** | Bearer token exposure | Short-lived tokens via Azure Managed Identity in prod |
| **SQL Injection** | Review queue queries | All queries use parameterised statements ✅ |
| **Rate Limiting** | LLM burst from batch processing | 300ms delay between events in `process_batch()` ✅ |
| **Audit Trail** | Untracked auto-created requests | All outcomes logged to `pipeline_outcomes.jsonl` ✅ |
| **LLM Audit** | Untracked LLM spend | All calls logged to `llm_metrics.jsonl` + Tab 6 ✅ |
| **Human Override** | LLM error propagation | Partial/LLM always go to review queue ✅ |
| **Data Privacy** | Asset/sensor PII in logs | Redact PII fields before writing to training data |

---

## 11. Deployment Guide

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Azure keys

# 3. Generate test + training data
python3 data/training/generate_sensor_training.py
python3 data/test/generate_test_data.py
python3 data/test/make_demo_csv.py

# 4. Launch dashboard
/Users/$USER/Library/Python/3.9/bin/streamlit run app.py --server.port 8502
```

### Demo Quickstart (no API keys needed)
1. Open http://localhost:8502
2. **Upload & Map tab** → Click "🎯 Load Demo Results"
3. **Sensor Monitor tab** → Click "🚀 Run Demo Sensor Events"  
   Or use the scenario picker to submit individual scenarios
4. **Review Queue tab** → Approve/Reject pending items
5. **Dashboard tab** → View KPIs and charts
6. **AI Assistant tab** → Ask *"Which assets have no match?"*
7. **LLM Analytics tab** → View model usage, latency, and token breakdown

### Production Architecture (Azure)
```
IoT Sensors
    ↓
Azure IoT Hub (ingestion)
    ↓
Azure Event Hubs (streaming)
    ↓
Azure Functions (sensor_ingestor + service_classifier)
    ↓
Azure Service Bus (review queue)    Azure CosmosDB (outcomes)
    ↓
IFM Hub REST API
    ↓
Azure Static Web Apps + FastAPI (dashboard backend)
```

| Current (Demo) | Production Replacement |
|---|---|
| Local file ingestion | Azure IoT Hub / Event Hubs |
| SQLite review queue | Azure Service Bus + CosmosDB |
| httpx mock | Real IFM Hub API endpoint |
| CSV training data | Azure ML fine-tuning pipeline |
| Streamlit | Azure Static Web Apps + FastAPI |


> Version 1.0 | May 2025 | Hackathon Edition

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Design (HLD)](#2-high-level-design-hld)
3. [Low-Level Design (LLD)](#3-low-level-design-lld)
4. [End-to-End Flow Diagrams](#4-end-to-end-flow-diagrams)
5. [Component Reference](#5-component-reference)
6. [Tech Stack](#6-tech-stack)
7. [Data Models](#7-data-models)
8. [IFM Hub API Contract](#8-ifm-hub-api-contract)
9. [Training Pipeline & Active Learning](#9-training-pipeline--active-learning)
10. [Security & Production Considerations](#10-security--production-considerations)
11. [Deployment Guide](#11-deployment-guide)

---

## 1. Executive Summary

The platform solves two connected problems:

| Problem | Solution |
|---|---|
| **Asset Mapping** — SFM assets need to be matched to IFM Hub assets | 5-approach fuzzy + LLM cascade (existing) |
| **Sensor → Work Order** — When a sensor alerts on an asset, the right IFM service classification must be found, and a work order auto-created (or queued for review) | New 4-tier classification pipeline + work order creator |

### Decision Rules (applies to both problems)

| Tier | Confidence | Action |
|---|---|---|
| 🟢 **Perfect Match** | ≥ 85 % | Auto-create work order in IFM Hub |
| 🟡 **Partial Match** | 50 – 84 % | Enqueue for human review |
| 🔵 **LLM Reasoned** | any | Enqueue for human review with AI explanation |
| 🔴 **No Match** | 0 % | Log only — no request created |

---

## 2. High-Level Design (HLD)

```mermaid
graph TB
    subgraph SENSORS["🌡️ Sensor Layer"]
        S1[IoT Sensor — HVAC]
        S2[IoT Sensor — Electrical]
        S3[IoT Sensor — Plumbing]
        S4[IoT Sensor — Refrigeration]
    end

    subgraph INGESTION["📡 Ingestion Layer"]
        IH["Azure IoT Hub<br/>(Production)<br/>— or —<br/>Webhook / MQTT<br/>(Demo)"]
        SI[SensorIngestor<br/>sensor_ingestor.py]
    end

    subgraph CLASSIFICATION["🧠 Classification Engine"]
        SC[ServiceClassifier<br/>service_classifier.py]
        LLM_SC[LLM Classification Agent<br/>service_classification_agent.py]
        CAT[(Service Catalog<br/>service_catalog.json<br/>20 classifications)]
    end

    subgraph DECISION["⚡ Decision Engine"]
        WOC[WorkOrderCreator<br/>work_order_creator.py]
        RQ[(Review Queue<br/>SQLite → Azure Service Bus)]
    end

    subgraph IFM["🏢 IFM Hub"]
        API[IFM Hub REST API<br/>POST /requests]
        WO[(Work Orders DB)]
    end

    subgraph ASSET_MAP["🗺️ Asset Mapping (existing)"]
        MATCH[LangGraph Matcher<br/>5-approach cascade]
        SFM_DB[(SFM Assets)]
        IFM_DB[(IFM Assets)]
    end

    subgraph DASHBOARD["📊 Streamlit Dashboard"]
        T1[Tab 1: Upload & Map]
        T2[Tab 2: Dashboard]
        T3[Tab 3: AI Assistant]
        T4[Tab 4: Sensor Monitor]
        T5[Tab 5: Review Queue]
    end

    S1 & S2 & S3 & S4 --> IH
    IH --> SI
    SI --> SC
    SC -- "Fuzzy < 50%" --> LLM_SC
    CAT --> SC
    SC -- "Perfect ≥85%" --> WOC
    SC -- "Partial 50-84%" --> RQ
    LLM_SC -- "LLM Reasoned" --> RQ
    SC -- "No Match" --> DASHBOARD
    WOC --> API
    API --> WO
    RQ --> T5
    WO --> T4
    MATCH --> T1 & T2
    SFM_DB & IFM_DB --> MATCH
```

---

## 3. Low-Level Design (LLD)

### 3.1 Service Classification Engine

```mermaid
flowchart TD
    A([Sensor Event arrives]) --> B{asset_type +\nalert_type known\nin catalog?}
    
    B -- Yes --> C[Score all 20 classifications\nweighted: asset_type 45%\nalert_type 40%\nkeywords 15%]
    B -- No  --> C

    C --> D{Best score?}
    D -- "≥ 85%\nPerfect" --> E[✅ Return Perfect Match\n+ classification ID]
    D -- "50–84%\nPartial" --> F[⚠️ Return Partial Match\n+ human review flag]
    D -- "30–49%\nAmbiguous" --> G[🤖 Call LLM Agent\nTop 5 candidates sent]
    D -- "< 30%\nUnknown" --> H[❌ No Match]

    G --> I{LLM response\nvalid?}
    I -- Yes --> J[🔵 Return LLM Reasoned\n+ AI reasoning text]
    I -- No  --> H
```

### 3.2 Work Order Decision Flow

```mermaid
flowchart TD
    A([Classification Result]) --> B{match_type?}
    
    B -- "Perfect Match\n≥85%" --> C[Build IFM Hub\nrequest payload]
    C --> D[POST /requests\nto IFM Hub API]
    D --> E[(Audit log\npipeline_outcomes.jsonl)]
    D --> F[✅ Work Order Created\nrequest_id returned]

    B -- "Partial Match\nor LLM Reasoned" --> G[Build request payload]
    G --> H[(Review Queue\nSQLite)]
    H --> I[⚠️ Awaits human review\nin Review Queue tab]
    I --> J{Operator decision}
    J -- Approve --> K[POST /requests\nto IFM Hub API]
    J -- Reject  --> L[❌ Logged\nno request]
    J -- Escalate --> M[🔺 Senior reviewer]
    K --> N[(Active learning log\nactive_learning.jsonl)]

    B -- "No Match" --> O[❌ Log only\nno request created]
```

### 3.3 Asset Mapping LangGraph Pipeline (existing)

```mermaid
flowchart LR
    START([START]) --> A1[node_approach1\nName + Make/Model\n+ Serial + Location]
    A1 -->|match?| END1([END])
    A1 -->|no match| A2[node_approach2\nName + Location\n+ Building]
    A2 -->|match?| END2([END])
    A2 -->|no match| A3[node_approach3\nMake/Model/Serial\n+ Location + Building]
    A3 -->|match?| END3([END])
    A3 -->|no match| P1[node_partial1\nName 50% + Location\n+ Building]
    P1 -->|match?| END4([END])
    P1 -->|no match| P2[node_partial2\nName 50% + Make/Model\n+ Location + Building]
    P2 -->|match?| END5([END])
    P2 -->|no match| LLM[node_llm_reason\nGPT-5.5 JSON mode]
    LLM --> FIN[node_finalize\nNo Match]
    FIN --> END6([END])
```

---

## 4. End-to-End Flow Diagrams

### 4.1 Happy Path — Auto Work Order Creation

```mermaid
sequenceDiagram
    participant Sensor as 🌡️ IoT Sensor
    participant Ingestor as SensorIngestor
    participant Classifier as ServiceClassifier
    participant Catalog as ServiceCatalog
    participant Creator as WorkOrderCreator
    participant IFM as IFM Hub API
    participant Dashboard as Streamlit UI

    Sensor->>Ingestor: POST {asset_type:"AHU", alert:"temperature_high", reading:78.5°F}
    Ingestor->>Ingestor: parse() + enrich() → SensorEvent
    Ingestor->>Classifier: classify(SensorEvent)
    Classifier->>Catalog: score all 20 classifications
    Catalog-->>Classifier: best=HVAC Cooling (score=92%)
    Classifier-->>Ingestor: {match_type:"Perfect", confidence:92%, sc_id:"4aa8..."}
    Ingestor->>Creator: process_event(event, classification)
    Creator->>Creator: decide_action() → AUTO_CREATE
    Creator->>Creator: build_ifm_request_payload()
    Creator->>IFM: POST /requests {serviceClassificationId, description, locationId, ...}
    IFM-->>Creator: {status:"created", request_id:"req-abc"}
    Creator-->>Dashboard: outcome {decision:"AUTO_CREATE", request_id:"req-abc"}
    Dashboard->>Dashboard: Update Sensor Monitor tab
```

### 4.2 Partial Match — Human Review Path

```mermaid
sequenceDiagram
    participant Sensor as 🌡️ IoT Sensor
    participant Classifier as ServiceClassifier
    participant Queue as ReviewQueue
    participant Operator as 👤 Human Operator
    participant IFM as IFM Hub API

    Sensor->>Classifier: {asset_type:"Pump", alert:"vibration_high"}
    Classifier-->>Queue: {match_type:"Partial", confidence:58%, sc_name:"Mechanical - Vibration"}
    Queue->>Queue: INSERT INTO review_items STATUS=PENDING
    Queue-->>Operator: Review Queue tab shows new item
    Operator->>Queue: approve(item_id, reviewer="John", notes="Confirmed bearing issue")
    Queue->>IFM: POST /requests (same payload, now approved)
    IFM-->>Queue: {status:"created", request_id:"req-xyz"}
    Queue->>Queue: UPDATE status=APPROVED
    Queue->>Queue: Append to active_learning.jsonl (label=approved)
```

### 4.3 LLM Reasoning Path

```mermaid
sequenceDiagram
    participant Sensor as 🌡️ IoT Sensor
    participant Classifier as ServiceClassifier
    participant LLM as GPT-5.5 Agent
    participant Queue as ReviewQueue

    Sensor->>Classifier: {asset_type:"BMS Controller", alert:"network_timeout"}
    Classifier->>Classifier: fuzzy score = 38% (below partial threshold)
    Classifier->>LLM: {asset_type, alert, top_5_candidates}
    LLM-->>Classifier: {chosen_id:"f559h6d3...", confidence:72, reasoning:"BMS network timeout maps to BAS Control System Fault..."}
    Classifier-->>Queue: {match_type:"LLM Reasoned", confidence:72%, reasoning:...}
    Queue->>Queue: INSERT PENDING with AI reasoning
    Note over Queue: Operator sees AI reasoning\nto aid approval decision
```

---

## 5. Component Reference

### Files Created / Modified

| File | Purpose | Status |
|---|---|---|
| `pipeline/sensor_ingestor.py` | Sensor event model + ingestion | ✅ New |
| `pipeline/service_classifier.py` | 4-tier classification engine | ✅ New |
| `pipeline/work_order_creator.py` | IFM Hub API client + decision engine | ✅ New |
| `pipeline/review_queue.py` | SQLite review queue | ✅ New |
| `pipeline/orchestrator.py` | E2E pipeline orchestration | ✅ New |
| `pipeline/matcher.py` | 5-approach fuzzy asset matcher | ✅ Existing |
| `pipeline/langgraph_agent.py` | LangGraph cascading matcher | ✅ Existing |
| `llm/service_classification_agent.py` | LLM classification agent | ✅ New |
| `llm/chat_agent.py` | LangChain ReAct chat agent | ✅ Existing |
| `data/service_catalog.json` | 20 IFM service classifications | ✅ New |
| `data/training/sensor_training_data.csv` | 1200-row labeled dataset | ✅ Generated |
| `data/training/sensor_training_data.jsonl` | JSONL format for LLM training | ✅ Generated |
| `data/training/active_learning.jsonl` | Approved/rejected review labels | ✅ Auto-generated |
| `data/review_queue.db` | SQLite review queue database | ✅ Auto-created |
| `app.py` | Streamlit dashboard (5 tabs) | ✅ Updated |

---

## 6. Tech Stack

### Core AI/ML
| Component | Technology | Purpose |
|---|---|---|
| LLM (Primary) | **Azure OpenAI GPT-5.5** | Asset matching + service classification |
| LLM (Secondary) | **Azure Anthropic Claude Sonnet 4.6** | Chat assistant |
| Agent Orchestration | **LangGraph** | Cascading match pipeline (DAG) |
| Chat Agent | **LangChain ReAct** | Natural language Q&A over results |
| Fuzzy Matching | **rapidfuzz** | String similarity (token_set_ratio) |

### Backend
| Component | Technology | Purpose |
|---|---|---|
| Language | **Python 3.9+** | All pipeline code |
| Event Model | **dataclasses** | Typed sensor events |
| Queue Storage | **SQLite** (demo) / **Azure Service Bus** (prod) | Review queue |
| HTTP Client | **httpx** | IFM Hub API calls |
| Config | **python-dotenv** | Secrets management |

### Frontend
| Component | Technology | Purpose |
|---|---|---|
| Dashboard | **Streamlit** | 5-tab interactive UI |
| Charts | **Plotly Express** | KPIs, donut, histograms, bar charts |

### Data
| Component | Technology | Purpose |
|---|---|---|
| Training Data | **pandas + CSV/JSONL** | 1200-row labeled sensor dataset |
| Asset Data | **Excel (openpyxl)** | SFM/IFM exports |
| Service Catalog | **JSON** | 20 IFM service classifications |

### Production (recommended upgrades)
| Current (Demo) | Production Replacement |
|---|---|
| Local file ingestion | Azure IoT Hub / Event Hubs |
| SQLite review queue | Azure Service Bus + CosmosDB |
| httpx mock | Real IFM Hub API endpoint |
| CSV training data | Azure ML fine-tuning pipeline |
| Streamlit | Azure Static Web Apps + FastAPI backend |

---

## 7. Data Models

### SensorEvent
```python
@dataclass
class SensorEvent:
    event_id:    str       # UUID
    sensor_id:   str       # Physical sensor ID
    asset_id:    str       # SFM asset reference
    asset_name:  str       # Human-readable name
    asset_type:  str       # AHU / Chiller / Pump / etc.
    alert_type:  str       # temperature_high / power_failure / etc.
    severity:    str       # INFO / LOW / MEDIUM / HIGH / CRITICAL
    reading:     SensorReading   # value, unit, threshold_min, threshold_max
    location_id: str       # IFM Hub location UUID
    building:    str
    floor:       str
    room:        str
    timestamp:   str       # ISO 8601
```

### ClassificationResult
```python
{
    "event_id":                    str,
    "asset_id":                    str,
    "location_id":                 str,
    "service_classification_id":   str,    # IFM Hub SC UUID
    "service_classification_name": str,
    "category":                    str,    # HVAC / Electrical / Plumbing / ...
    "subcategory":                 str,
    "priority":                    str,    # CRITICAL / HIGH / MEDIUM / LOW
    "sla_hours":                   int,
    "confidence":                  float,  # 0–100
    "match_type":                  str,    # Perfect / Partial / LLM Reasoned / No Match
    "reasoning":                   str,    # LLM or fuzzy explanation
    "auto_create_threshold":       float,  # 85
}
```

### IFM Hub Request Payload
```json
{
    "orgs":                         ["uuid1", "uuid2"],
    "tenantId":                     "601205a7f110dd542d9237bc",
    "id":                           "{{requestId}}",
    "reportedDate":                 "2025-05-04T08:00:00",
    "alternateId":                  "AI-ABC12345",
    "description":                  "Auto-generated: AHU-1 | temperature_high | 78.5°F (threshold: 72°F)",
    "locationId":                   "f754334d-17cc-4890-bc58-2a4e1a386549",
    "serviceClassificationId":      "4aa86b28-c506-11ed-afa1-0242ac120002",
    "relatedServiceClassificationId": [],
    "requestorId":                  "cb0795b9-32d6-4574-8a88-bd8fc5b1b5cd",
    "ownerId":                      "cb0795b9-32d6-4574-8a88-bd8fc5b1b5cd",
    "source":                       "AI Sensor Alert",
    "sourceApp":                    "sfm-ai-platform",
    "statusId":                     "13ef1492-8e5f-4337-9751-c42d1a823edf",
    "attachments":                  [],
    "_meta": {
        "sensor_event_id":          "uuid",
        "match_type":               "Perfect Match",
        "match_confidence":         92.0,
        "ai_reasoning":             "..."
    }
}
```

---

## 8. IFM Hub API Contract

### Create Work Order Request
```
POST {IFM_BASE_URL}/requests
Authorization: Bearer {IFM_API_KEY}
Content-Type: application/json
Body: <IFM Hub Request Payload above>
```

### Environment Variables (`.env`)
```ini
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://admv-mogidbp0-eastus2.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.5_1
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Anthropic (Claude)
AZURE_ANTHROPIC_ENDPOINT=https://admv-mogidbp0-eastus2.services.ai.azure.com/anthropic/
AZURE_ANTHROPIC_API_KEY=<key>

# IFM Hub
IFM_BASE_URL=https://api.ifm-hub.example.com
IFM_API_KEY=<key>
IFM_TENANT_ID=601205a7f110dd542d9237bc
IFM_ORG_IDS=["7bb1b889-...", "fea0683c-..."]
IFM_REQUESTOR_ID=cb0795b9-32d6-4574-8a88-bd8fc5b1b5cd

# Pipeline thresholds
PERFECT_THRESHOLD=85
```

---

## 9. Training Pipeline & Active Learning

```
data/training/
├── training_examples.json          ← 13 asset-mapping examples (SFM↔IFM)
├── sensor_training_data.csv        ← 1200 sensor classification rows
├── sensor_training_data.jsonl      ← Same, JSONL for LLM fine-tuning
├── sensor_training_data.json       ← First 150 rows as few-shot examples
└── active_learning.jsonl           ← Operator approvals/rejections (grows over time)
```

### Active Learning Loop

```mermaid
flowchart LR
    A[Sensor Event] --> B[Classification]
    B --> C{Partial/LLM?}
    C -- Yes --> D[Review Queue]
    D --> E[Operator Reviews]
    E -- Approve --> F[Work Order Created]
    E -- Reject  --> G[Discarded]
    F & G --> H[(active_learning.jsonl\nlabel=approved/rejected)]
    H --> I[Periodic fine-tuning\nor prompt improvement]
    I --> B
```

### Training Data Distribution (1200 rows)
| Match Type | Count | % |
|---|---|---|
| Perfect Match | 569 | 47.4% |
| LLM Reasoned | 232 | 19.3% |
| Partial Match | 210 | 17.5% |
| No Match | 189 | 15.8% |

---

## 10. Security & Production Considerations

| Area | Concern | Mitigation |
|---|---|---|
| **API Keys** | Hardcoded credentials | Move all keys to `.env` / Azure Key Vault |
| **Input Validation** | Malicious sensor payloads | Schema validation on all SensorEvent fields |
| **IFM API Auth** | Bearer token exposure | Use short-lived tokens via Azure Managed Identity |
| **SQL Injection** | Review queue SQLite queries | All queries use parameterised statements ✅ |
| **Rate Limiting** | LLM / IFM API overload | Implement exponential backoff + circuit breaker |
| **Audit Trail** | Untracked auto-created requests | All outcomes logged to `pipeline_outcomes.jsonl` ✅ |
| **Human Override** | LLM error propagation | Partial/LLM always go to review queue ✅ |
| **Data Privacy** | Asset/sensor PII in logs | Redact PII fields before writing to training data |

---

## 11. Deployment Guide

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Azure keys

# 3. Generate training data
python3 data/training/generate_sensor_training.py

# 4. Launch dashboard
streamlit run app.py --server.port 8501
```

### Demo Quickstart (no API keys needed)
1. Open http://localhost:8501
2. **Upload & Map tab** → Click "🎯 Load Demo Results"
3. **Sensor Monitor tab** → Click "🚀 Run Demo Sensor Events"
4. **Review Queue tab** → Approve/Reject pending items
5. **Dashboard tab** → View KPIs and charts
6. **AI Assistant tab** → Ask "Which assets have no match?"

### Production Architecture (Azure)
```
IoT Sensors
    ↓
Azure IoT Hub (ingestion)
    ↓
Azure Event Hubs (streaming)
    ↓
Azure Functions (sensor_ingestor + service_classifier)
    ↓
Azure Service Bus (review queue)
    ↓
Azure CosmosDB (outcomes + audit log)
    ↓
Azure App Service (Streamlit / FastAPI dashboard)
    ↓
IFM Hub REST API (work order creation)
```

---

*Generated by GitHub Copilot — SFM ↔ IFM AI Platform*
