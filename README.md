# 🛃 Customs Trade Classification — OpenEnv Environment

> **A real-world autonomous compliance pipeline for the Meta PyTorch OpenEnv Hackathon.**
>
> This environment places an AI agent inside a live customs brokerage workflow. The agent must classify international shipments against the **US Harmonized Tariff Schedule (HTS)**, verify country-of-origin against an **OFAC sanctions database**, and submit legally binding `approve` or `hold` verdicts — all within a strict step budget. It is designed to expose the exact cognitive boundaries of frontier language models: tabular data navigation, geopolitical deductive reasoning, and hierarchical code disambiguation.

---

## ✨ Key Features

- **Live HTS Lookup Tool** — The agent is never given a tariff code. It must call `lookup_hs` with progressively narrower prefixes to navigate 10,000+ rows of the real US tariff schedule, mimicking how a human specialist actually works.
- **Partial-Credit Fuzzy Grading** — Powered by `rapidfuzz`, the subheading grader awards graduated credit based on string similarity. A near-miss on a 10-digit code still earns partial reward, creating a dense, continuous gradient for RL training signal.
- **Strict Normalized Reward** — Total episode reward is normalized to `[0.0, 1.0]` with six independently graded components (Chapter, Heading, Subheading, Duty Rate, Sanctions Check, Final Verdict). Every step contributes to the signal — there is no sparse terminal reward.
- **One-Way State Machine** — The environment enforces a forward-only workflow. Once a chapter is confirmed, the agent cannot backtrack. This prevents reward hacking and forces genuine multi-step planning.
- **Concurrent Session Support** — `SUPPORTS_CONCURRENT_SESSIONS = True`. Fully safe for parallel evaluation runs.

---

## 🎯 Reward Structure

| Component | Weight | Grader Type |
|-----------|--------|------------|
| Chapter Classification | 15% | Exact match |
| Heading Classification | 20% | Exact match + partial credit |
| Subheading Classification | 25% | `rapidfuzz` similarity ratio |
| Duty Rate | 20% | Normalized float comparison |
| Sanctions Check | 10% | Deterministic OFAC lookup |
| Final Verdict (approve/hold) | 10% | Boolean |

---

## 🗂️ Task Breakdown

### 🟢 `task_easy` — Standard Shipment Classification
**Baseline Target: ≥ 1.00** | Max Steps: 15

The agent receives a shipment with a clear, unambiguous product description containing an explicit `[SYSTEM HINT]` pointing to the correct HS Chapter and Heading. The country of origin is a non-sanctioned nation.

**What it tests:** Basic tool-use discipline. Can the agent follow the `lookup_hs → classify_chapter → classify_heading → classify_subheading → check_duty → lookup_sanctions → submit` workflow without deviating?

**Example shipment:** *Polypropylene buckets, injection-moulded, 10L capacity. Country of manufacture: Germany.*

---

### 🟡 `task_medium` — Sanctions & Geopolitical Deductive Reasoning
**Baseline Target: ~0.59** | Max Steps: 20

The agent receives dual-use industrial goods (centrifuges, heat exchangers, hydraulic presses) with obfuscated procurement descriptions. The country of origin is a **sanctioned nation** (Iran, Russia, North Korea, Syria, Belarus) stated explicitly in the product text. The agent must read the description, deduce the geopolitical risk, call `lookup_sanctions`, and correctly submit a `hold` verdict.

**What it tests:** Multi-step deductive reasoning. The agent must connect unstructured text signals to a structured database lookup and reach the correct legal conclusion.

**Example shipment:**
> *"Heat exchanger, shell and tube type ... Stated use: process fluid cooling in chemical plant. Procured by 'Global Horizon Trading'. Origin: North Korea."*

---

### 🔴 `task_hard` — Hierarchical Tabular Math & Threshold Disambiguation
**Baseline Target: ~0.35** | Max Steps: 30

The agent receives highly technical industrial equipment (lyophilizers, mass spectrometers, industrial microwave arrays) with **no geographic hints**. The country of origin is a benign, safe nation (Japan, Canada, Australia) — so the agent cannot earn sanctions points via guessing. To score, it must navigate the HTS table through **technical parameter thresholds**: rated wattage, condenser capacity, operating temperature ranges, and equipment classification boundaries that separate adjacent 10-digit subheadings.

**What it tests:** Continuous, hierarchical tabular data navigation under uncertainty. The agent must distinguish between adjacent HTS codes separated only by a single technical parameter (e.g., `>= 6kW` vs `< 6kW` for industrial microwave ovens).

**Example shipment:**
> *"Microwave oven industrial heating unit, 2.45 GHz magnetron array, continuous conveyor type, rated power input 6 kW ... [SYSTEM HINT: The correct HS Heading is 8514. Pay close attention to whether it is domestic or industrial/laboratory use.]*"

---

## 📊 Verified Baseline Scores

> [!IMPORTANT]
> **Note on Difficulty Calibration: Human difficulty does not equal LLM difficulty.**
>
> During our baseline testing with `meta-llama/Llama-3.3-70B-Instruct`, we discovered that semantic models actually excel at the deductive reasoning required for geopolitical sanctions checks (our Medium tier). Conversely, the true "Hard" boundary for frontier models is navigating continuous, hierarchical tabular data and strict mathematical thresholding (HTS subheadings). Therefore, our difficulty curve is **explicitly calibrated to Machine Learning cognitive limits, not human intuition**, resulting in a strictly verified `1.00 → 0.59 → 0.35` baseline staircase.

| Model | `task_easy` | `task_medium` | `task_hard` |
|-------|------------|--------------|------------|
| `meta-llama/Llama-3.3-70B-Instruct` | **1.00** | **0.59** | **0.35** |
| `qwen/qwen3-next-80b-a3b-instruct` | 1.00 | 0.97 | 0.62 |
| `Qwen/Qwen2.5-72B-Instruct` | 0.00* | 0.10* | 0.10* |

> \* `Qwen2.5-72B` failed due to a `404` endpoint error on the Nvidia NIM router during our test window — not an environment bug.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for container deployment)
- An [OpenEnv](https://openenv.dev) installation or a valid `HF_TOKEN`

### Run Locally with OpenEnv

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/customs-trade-classification
cd customs-trade-classification

# Install dependencies
pip install -r requirements.txt

# Validate the environment manifest
openenv validate

# Run a single task episode
openenv run task_easy
```

### Run the Inference Baseline Script

```bash
# Set your credentials
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

# Start the environment server
uvicorn server.app:app --host 127.0.0.1 --port 7860 &

# Run the full three-task evaluation
python inference.py
```

### Run with Docker

```bash
docker build -t customs-trade-classification .
docker run -p 7860:7860 customs-trade-classification
```

---

## 🏗️ Project Structure

```
customs-trade-classification/
│
├── openenv.yaml               # Environment manifest — tasks, difficulty, reward thresholds
├── Dockerfile                 # Production container (python:3.11-slim, port 7860)
├── pyproject.toml             # Package metadata and build configuration
├── requirements.txt           # Pinned dependencies for reproducible builds
│
├── inference.py               # Official evaluation runner — outputs [START]/[STEP]/[END] logs
├── test_local.py              # Local test harness mirroring inference.py
├── client.py                  # OpenEnv API client with full Pydantic model parsing
├── models.py                  # CustomsState and CustomsAction Pydantic models
│
├── server/
│   ├── app.py                 # FastAPI application entry point (port 7860)
│   ├── environment.py         # Core RL environment — state machine, graders, shipment catalogue
│   └── customs_env_environment.py  # BaseEnvironment wrapper for OpenEnv compatibility
│
├── data/
│   ├── hts_data.json          # HTS tariff schedule (subset, ~5,000 entries)
│   ├── ofac_sdn.csv           # OFAC sanctions country list
│   └── loader.py              # Data parsing and lookup index construction
│
└── tests/
    └── test_graders.py        # pytest suite validating grader math and fuzzy scoring logic
```

---

## 🧠 Environment Architecture

```
Agent
  │
  ▼  action (JSON)
┌─────────────────────────────────┐
│         FastAPI Server          │  ◄─── openenv.yaml manifest
│  POST /step   POST /reset       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      CustomsEnvironment         │
│                                 │
│  ┌─────────────────────────┐   │
│  │     CustomsState        │   │ ◄── Pydantic model, 18 fields
│  │  chapter_score: float   │   │     session-isolated per reset()
│  │  heading_score: float   │   │
│  │  sanctions_result: str  │   │
│  │  step_count: int        │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌──────────┐ ┌─────────────┐  │
│  │ HTS DB   │ │ OFAC SDN DB │  │ ◄── Loaded once at startup
│  └──────────┘ └─────────────┘  │
└─────────────────────────────────┘
             │
             ▼
  Observation (JSON) → Agent
```

---

## 📋 Action Space

| Action | Value Format | When Available |
|--------|-------------|----------------|
| `lookup_hs` | HTS prefix string (e.g. `"84"`, `"8419"`) | Always (lookup steps) |
| `classify_chapter` | 2-digit string (e.g. `"84"`) | Before chapter confirmed |
| `classify_heading` | 4-digit string (e.g. `"8419"`) | After chapter confirmed |
| `classify_subheading` | 10-digit string (e.g. `"8419.39.02.00"`) | After heading confirmed |
| `check_duty` | Duty rate string (e.g. `"Free"`, `"3.5%"`) | After subheading confirmed |
| `lookup_sanctions` | Country name string (e.g. `"Iran"`) | After duty confirmed |
| `check_sanctions` | `"flagged"` or `"clear"` | After sanctions lookup |
| `submit` | `"approve"` or `"hold"` | Final step only |

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026. Environment calibrated against `meta-llama/Llama-3.3-70B-Instruct`.*
