# TruthLens AI — Multimodal Misinformation Verification & Explainability System

TruthLens AI is an end-to-end multimodal misinformation detection platform that evaluates news claims across **Text, Image, URL, Audio, and Video** modalities. It utilizes task-specific encoders, Cross-Modal Reliability-Aware Evidence Verification (RAEV), adaptive late fusion with XGBoost, and SHAP (SHapley Additive exPlanations) for human-interpretable factor attribution.

---

## Architecture Overview

```
                                  USER INPUT
               [Text]    [Image]    [URL]    [Audio]    [Video]
                 │          │         │         │          │
                 ▼          ▼         ▼         ▼          ▼
             RoBERTa       ViT      Domain   Whisper    Uniform
             768-dim     768-dim   Signals  + Energy    Sampling
                 │          │         │         │          │
                 ▼          ▼         ▼         ▼          ▼
            ────────────── Modality Reliability ─────────────
                                    │
                                    ▼
                RAEV (Reliability-Aware Evidence Verification)
                - Pairwise Cross-Modal Agreement Matrix
                - RoBERTa-MNLI Entailment / Contradiction
                - Overall Reliability & Evidence Status (HIGH / MED / LOW)
                                    │
                                    ▼
                 Reliability-Aware Adaptive Fusion (786-dim)
                                    │
                                    ▼
                 Calibrated Classifier & XGBoost Model
                                    │
                                    ▼
                     VERDICT: REAL / FAKE / MISLEADING / UNVERIFIED / INSUFFICIENT EVIDENCE
                     (Confidence + Calibrated Probabilities + Legacy Prediction Alias)
                                    │
                                    ▼
                 SHAP TreeExplainer Attribution Factors
                                    │
                                    ▼
               PostgreSQL / Resilient SQLite Audit History
                                    │
                                    ▼
                       React 19 Interactive Web UI
```

---

## Features & Highlights

1. **True Multimodal Coverage**: Single-modality and arbitrary combinations of Text, Image, Source URL, Audio recordings, and Video files.
2. **Missing Modality Robustness**: System handles omitted modalities gracefully using reliability masking (missing $\neq$ unreliable).
3. **Evidence Verification & Agreement**: Computes cross-modal consistency (e.g. Text vs Image) and pairwise agreement between all available modalities.
4. **No Placeholders / No Hardcoding**: Real model inference using RoBERTa, ViT, Whisper-tiny, and XGBoost; verified against benchmark datasets.
5. **SHAP Factor Attribution**: Generates quantitative supporting and contradictory factors driving each decision.
6. **Dual-Dialect Database Persistence**: Native PostgreSQL storage with automatic, resilient SQLite fallback if PostgreSQL is not running.
7. **Modern React 19 Frontend**: Dark-themed dashboard, live multi-modal verification workspace, searchable audit history, and real dataset evaluation bench.

---

## Directory Structure

```
MEGA_PROJECT_1/
├── alembic/                  # Database migration scripts
├── controllers/              # Pipeline controller & orchestration
├── datasets/                 # PolitiFact, GossipCop, and cleaned text datasets
├── frontend/                 # React 19 + TypeScript + Vite + Tailwind UI
│   ├── src/
│   │   ├── components/       # TopBar, Layout, Cards
│   │   ├── pages/            # Dashboard, Verify, History, Evaluation
│   │   └── types/            # TypeScript interfaces
├── models/                   # Serialized classifiers & scalers
├── modules/
│   ├── adaptive_fusion/      # Reliability fusion & XGBoost classifier
│   ├── explainability/       # SHAP TreeExplainer integration
│   ├── raev/                 # Reliability-Aware Evidence Verification
│   └── v2/                   # V2 Encoders (RoBERTa, ViT, URL Intelligence)
├── services/
│   ├── verification_service.py # Core verification singleton
│   └── evaluation_service.py   # Labeled benchmark evaluator
├── storage/                  # SQLite fallback database & persistent features
├── tests/                    # Pytest test suite (comprehensive V1 & V2)
├── uploads/                  # Temporary media upload storage
├── .env                      # Environment configuration
├── alembic.ini               # Alembic configuration
├── requirement.txt           # Python production dependencies
└── server.py                 # FastAPI backend server
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- PostgreSQL (Optional — system falls back automatically to SQLite `storage/truthlens.db`)

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirement.txt

# (Optional) Run Alembic migrations for PostgreSQL
alembic upgrade head
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run build
```

---

### 1. One-Click Launch (Recommended)
On Windows, you can start both Backend and Frontend together using PowerShell or Command Prompt:

```powershell
# PowerShell (starts both Backend and Frontend in separate windows)
.\run.ps1

# Options:
# .\run.ps1 -Backend    # Start only FastAPI Backend
# .\run.ps1 -Frontend   # Start only React Frontend
# .\run.ps1 -Streamlit  # Start Streamlit App
# .\run.ps1 -Test       # Run test suite
```

Alternatively, run `start.bat` or `start_streamlit.bat`.

### 2. Manual Start

#### Backend API Server
```bash
python server.py
# Server runs at: http://localhost:8000
# OpenAPI Swagger docs at: http://localhost:8000/docs
```

#### Frontend Web Interface (React 19)
```bash
cd frontend
npm run dev
# Frontend runs at: http://localhost:5173
```

#### Streamlit Interactive Dashboard
```bash
streamlit run app.py
# Streamlit dashboard runs at: http://localhost:8501
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and model readiness |
| `POST` | `/api/verify/text` | Verify headline/claim text |
| `POST` | `/api/verify/image` | Verify image and optional claim text |
| `POST` | `/api/verify/url` | Verify news URL and domain reliability |
| `POST` | `/api/verify/audio` | Speech-to-text (Whisper) & acoustic reliability |
| `POST` | `/api/verify/video` | Uniform temporal frame sampling & visual consistency |
| `POST` | `/api/verify/multimodal` | Unified verification with any combination of modalities |
| `GET` | `/api/history` | Paginated verification history with search and verdict filters |
| `GET` | `/api/history/{id}` | Complete audit trail for a specific verification |
| `GET` | `/api/dashboard/stats` | Real-time statistics from verification database |
| `GET` | `/api/evaluation` | Evaluation benchmarks vs runtime statistics overview |
| `POST` | `/api/evaluation/run` | Run real benchmark evaluation against a labeled dataset |

---

## Training & Full Pipeline Orchestration

To reproduce the full TruthLens V2 model training and evaluation sequentially:

```bash
python run_full_pipeline.py
```

This master pipeline executes the following 5 stages:
1. **Journalistic Embeddings Generation** (`generate_journalistic_embeddings.py`): Checks and generates balanced 768-dim RoBERTa-base embeddings from PolitiFact and GossipCop datasets (3,200 samples).
2. **Text Head Retraining** (`retrain_text_head.py`): Retrains the RoBERTa classification head using combined multimodal and journalistic news features with balanced class weights to maximize REAL/FAKE recall.
3. **Image Head Retraining** (`train_image_model.py`): Retrains the ViT image classifier head.
4. **Multimodal Fusion Retraining** (`train_fusion_model.py`): Trains and evaluates fusion architectures (Reliability-Weighted Averaging, Gated Fusion, Cross-Modal Attention), saving the optimal calibrated model.
5. **Final Test Evaluation** (`evaluate_v2.py`): Runs complete validation against the frozen test set and generates benchmark metrics in `outputs/v2/`.

### Individual Model Retraining Scripts
- Text Classifier: `python retrain_text_head.py`
- Image Classifier: `python train_image_model.py`
- Fusion Classifier: `python train_fusion_model.py`

---

## Running Automated Tests

Run the full pytest suite (73/73 passing):
```bash
# Run all tests
python -m pytest tests/ -v

# Run the comprehensive API & backend test suite
python -m pytest tests/test_server_api.py -v

# Run V2 comprehensive tests
python -m pytest tests/test_v2_comprehensive.py -v
```

---

## Official Benchmark Results (Frozen Test Set, N=600)

| Metric | V1 Baseline | V2 Improved | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 76.00% | **82.83%** | **+6.83%** |
| **Recall** | 74.67% | **94.33%** | **+19.66%** |
| **F1-Score** | 75.68% | **84.60%** | **+8.92%** |
| **ROC-AUC** | 0.8661 | **0.9009** | **+0.0348** |
| **Brier Score** (lower is better) | 0.1894 | **0.1206** | **-0.0688** |
| **ECE** (lower is better) | 0.1660 | **0.0313** | **-0.1347** |

To re-run the benchmark evaluation:
```bash
python evaluate_v2.py
```
Outputs and plots are generated in `outputs/v2/`.
