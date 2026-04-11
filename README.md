---
title: MetricTrace-Gym
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---
# MetricTrace-Gym

> An OpenEnv-compliant RL environment where agents act as Sustainability Auditors, extracting and validating ESG metrics against the CSRD framework.

## 🌍 Domain Motivation

The global ESG compliance industry is valued at **$7B+** and growing rapidly. Companies must report Environmental, Social, and Governance metrics under frameworks like the **EU Corporate Sustainability Reporting Directive (CSRD)** and **ISSB S1/S2** standards. Manual auditing of multi-page sustainability reports is:

- **Error-prone**: Auditors must cross-reference narrative claims against tables scattered across 100+ page PDFs
- **Time-intensive**: A single audit can take 40–80 hours
- **Susceptible to greenwashing**: Companies may make misleading claims that contradict their own quantitative data

This environment trains RL agents to automate ESG report auditing — navigating document stores, extracting metrics from structured/unstructured tables, performing unit conversions, and detecting greenwashing discrepancies.

**Filling a Gap for the RL Community:** This project addresses a key missing piece for AI evaluation by providing complex, stateful, long-form PDF reasoning tasks. It requires agents to demonstrate memory, multi-page spatial navigation, and quantitative cross-referencing—all graded fully deterministically without slow or expensive "LLM-as-a-judge" mechanisms.

---

## 🏗️ Architecture

```
MetricTrace-Gym/
├── pyproject.toml         # OpenEnv component constraints & entrypoints
├── uv.lock                # Deterministic dependency resolution
├── models.py              # Pydantic Action, Observation, State models
├── client.py              # EnvClient HTTP wrapper
├── openenv.yaml           # OpenEnv manifest (spec_version: 1)
├── inference.py           # Baseline LLM agent with structured logging
├── server/
│   ├── environment.py     # Core RL environment logic & grader
│   └── app.py             # FastAPI create_app factory & UI
├── Dockerfile             # Unified root-level execution container
└── README.md              # This file
```

### 3-Component Design Pattern

| Component | File | Purpose |
|-----------|------|---------|
| **Typed Models** | `models.py` | Pydantic schemas for Action, Observation, State |
| **Server Logic** | `server/environment.py` | Stateful environment with `reset()`, `step()`, `state()` |
| **Client Wrapper** | `client.py` | HTTP client exposing Pythonic API |

---

## 🎯 Action Space

| Action | Arguments | Description |
|--------|-----------|-------------|
| `search_page` | `query?: str`, `page_number?: int` | Navigate to a page by keyword search or direct number |
| `extract_table` | `table_index: int = 0` | Extract structured table data from the current page |
| `submit_finding` | `metric_name: str`, `value: float`, `unit: str`, `evidence_pages?: [int]`, `discrepancy_detected?: bool`, `narrative_assessment?: str` | Submit an audited ESG finding for grading |

---

## 📊 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_page` | `PageContent?` | Content of the currently viewed page |
| `extracted_table` | `TableData?` | Most recently extracted table |
| `feedback` | `str` | Natural-language feedback on the last action |
| `task_description` | `str` | Current task objective |
| `task_difficulty` | `str` | `easy`, `medium`, or `hard` |
| `available_pages` | `int` | Total pages in document store |
| `steps_taken` | `int` | Steps used so far |
| `max_steps` | `int` | Maximum allowed steps |
| `score` | `float` | Normalized score (0.0–1.0) |
| `done` | `bool` | Whether the episode has ended |
| `error` | `str?` | Error message if last action was invalid |

---

## 📋 Tasks

### 🟢 Easy: Extract Total Water Consumption
- **Objective**: Find the explicitly labeled "Total Water Consumption" KPI from a structured table
- **Key skill**: Single-page navigation + table extraction
- **Max steps**: 10

### 🟡 Medium: Aggregate Scope 1 Emissions
- **Objective**: Calculate total Scope 1 GHG emissions by aggregating data from multiple tables
- **Key skills**: Multi-page navigation + unit conversion (kg CO2e → metric tons CO2e) + aggregation
- **Gotcha**: Fleet emissions reported in kg, must convert to metric tons
- **Max steps**: 15

### 🔴 Hard: Detect Greenwashing Discrepancy
- **Objective**: Cross-validate the CEO's claim of "10% emission reduction" against actual data
- **Key skills**: Narrative analysis + quantitative verification + discrepancy detection
- **The twist**: The CEO claims a 10% *reduction*, but the data shows a 3% *increase*
- **Max steps**: 20

---

## 💰 Reward Design

**Dense, potential-based Scalar rewards** with partial credit: Our environment applies scalar reward shaping so agents receive partial credit for sub-steps (e.g., finding the correct evidence page, extracting a table) even if the final metric submission fails.

| Component | Reward | Description |
|-----------|--------|-------------|
| Find evidence page | +0.10 | Navigating to a page containing ground-truth data |
| All evidence pages | +0.05 | Bonus when all required pages are visited |
| Extract relevant table | +0.15 | Extracting a table from an evidence page |
| Valid submission schema | +0.10 | Submitting a well-formed finding |
| Correct metric name | +0.05 | Fuzzy match on metric name |
| Correct unit | +0.10 | Unit equivalence check |
| Exact value match | +0.35 | Within tolerance of ground truth |
| Close value | +0.20 | Within 5% relative error |
| Discrepancy detection | +0.20 | Correct greenwashing flag (hard task) |
| Invalid action | −0.05 | Malformed arguments or out-of-range indices |

---

## 🔬 Grading

All grading is **deterministic and programmatic** (0.0–1.0):

- **No LLM-based grading** — scores are reproducible.
- **Greenwashing Determinism**: The Greenwashing grader compares the agent's claim directly against a hardcoded "ground-truth mismatch" embedded in the task data, ensuring 100% deterministic evaluation of narrative vs. quantitative contradiction.
- **Numeric comparison** with configurable tolerance (default 1–2%)
- **Fuzzy metric-name matching** via token overlap
- **Unit equivalence** mapping (e.g., "metric tons CO2e" ≡ "tCO2e")

## 🔄 Environment Design: Error Recovery

A key design strength of ESG-Audit-Gym is **structured error recovery**. If an agent makes a malformed tool call or invalid parameter request, the environment does not crash. Instead, it catches the exception and returns the exact error string (e.g., `Action conversion error: 'NoneType' object has no attribute 'get'`) back to the agent in the `Observation` payload. This allows the LLM to learn from its mistake, self-correct its JSON outputs, and retry—a vital feature for evaluating real-world reliability.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# This project uses `uv` for ultra-fast, strictly-locked dependency resolution
uv sync
```

### 2. Start the Environment Server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run the Baseline Agent

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export GROQ_API_KEY="gsk_..."
export ENV_BASE_URL="http://localhost:8000"

uv run python inference.py
```

### 4. Baseline Scores (`llama-3.3-70b-versatile`)

Our baseline reference agent successfully navigates the environment, including recovering from JSON generation errors:

- `easy_water_consumption`: **0.952** (3 steps)
- `medium_scope1_aggregation`: **0.870** (5 steps, successfully recovered from Action conversion error)
- `hard_greenwashing_detection`: **0.889** (5 steps)
- **Average Score**: **0.904**

### 5. Docker

```bash
docker build -t metrictrace-gym .
docker run -p 8000:8000 metrictrace-gym
```

---

## 📝 Logging Format

The inference script emits structured stdout lines:

```
[START] task=easy_water_consumption env=metrictrace-gym model=llama-3.3
[STEP] step=1 action=search_page reward=0.10 done=False error=null
[STEP] step=2 action=extract_table reward=0.15 done=False error=null
[STEP] step=3 action=submit_finding reward=0.60 done=True error=null
[END] task=easy_water_consumption success=True steps=3 score=0.810 rewards=0.10,0.15,0.60
```

---

## 🧪 Validation

```bash
openenv validate openenv.yaml
```

---

## 📚 Regulatory References

- **CSRD**: [EU Corporate Sustainability Reporting Directive](https://finance.ec.europa.eu/capital-markets-union-and-financial-markets/company-reporting-and-auditing/company-reporting/corporate-sustainability-reporting_en)
- **ISSB S1/S2**: [IFRS Sustainability Disclosure Standards](https://www.ifrs.org/issued-standards/ifrs-sustainability-standards-navigator/)
- **ESRS**: [European Sustainability Reporting Standards](https://www.efrag.org/)
- **GHG Protocol**: [Scope 1/2/3 emissions classification](https://ghgprotocol.org/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
