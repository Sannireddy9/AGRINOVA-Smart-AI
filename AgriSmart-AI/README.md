# AgriNova Smart AI — Intelligent Agriculture for a Sustainable Future



## 📌 Problem & Solution Overview

### The Problem
Smallholder farmers in India face compounding agronomic challenges: unpredictable weather anomalies due to climate change, sudden foliar disease outbreaks that devastate harvests, uncalibrated water usage leading to aquifer depletion or root rot, and soil degradation from imbalanced chemical fertilization. Traditional agricultural advisory systems often require expensive IoT hardware, provide opaque recommendations, or suffer from long turnaround times.

### The AgriSmart AI Solution
**AgriNova Smart AI** is a transparent, farmer-centric agronomy platform that delivers actionable, explainable, and accessible intelligence without requiring proprietary hardware or Internet-of-Things (IoT) field sensors:
1. **Foliar Disease Detection (Core)**: Computer vision pipeline built on deep transfer learning to detect crop pathologies early from standard smartphone leaf photographs.
2. **Crop Recommendation (Bonus A)**: Machine-learning tabular decision engine predicting optimal, climate-resilient crop varieties based on local soil chemistry and climate envelopes.
3. **Smart Irrigation Advisor (Bonus B)**: Explainable 5-tier heuristic irrigation scheduling engine combining farmer-observed soil moisture with multi-day precipitation forecasts.
4. **Weather-Based Intelligence (Bonus C)**: Real-time agrometeorological tracking via Open-Meteo providing 7-day outlooks, foliar disease risk indices, heat-stress alerts, and chemical spray drift windows.
5. **Farm Sustainability Score (Bonus D)**: Environmental stewardship index (0–100) featuring dynamic missing-data weight renormalization and empirical FAO / ICAR water-demand benchmarks.
6. **Farmer AI Assistant (Bonus E)**: Conversational decision support powered by Google Gemini (Free Tier) grounded strictly in active farm data, backed by a 100% deterministic local rule fallback engine.

---

## 🏗️ Architecture & Module Breakdown

AgriNovaSmart AI strictly distinguishes between **Computer Vision Machine Learning**, **Tabular Classification Machine Learning**, **Deterministic Expert Heuristic Systems**, and **Grounded Generative AI**.

```
AgriNova Smart-AI/
├── app/                              # Farmer-friendly web application (Flask)
│   ├── config.py                     # App configuration & environment loading
│   ├── main.py                       # App factory & route definitions
│   ├── routes/main.py                # Dashboard & API controllers
│   ├── services/                     # Service boundary layer (Disease, Crop, Weather, Irrigation, Sustainability, Assistant)
│   ├── static/                       # Custom responsive CSS & Vanilla JS
│   └── templates/                    # Modular Jinja2 HTML templates
├── data/                             # Data layer & model artifacts
│   ├── crop_recommendation/          # Crop dataset & lightweight 3.6 MB Random Forest artifact
│   ├── raw/                          # Raw image directory (.gitkeep, uncommitted)
│   ├── processed/                    # Processed class mapping (.gitkeep)
│   ├── sustainability/               # FAO / ICAR crop water benchmark reference data
│   └── uploads/                      # Temporary inference upload folder (.gitkeep)
├── model/                            # ML pipelines & decision logic
│   ├── config.py                     # Core vision hyperparameters & single source of truth
│   ├── dataset.py                    # Stratified dataset split & PyTorch DataLoaders
│   ├── evaluate.py                   # Macro-F1, per-class metrics & confusion matrix
│   ├── model.py                      # ResNet transfer learning architecture
│   ├── predict.py                    # Inference engine & documented CLI
│   ├── train.py                      # Training loop with early stopping & scheduler
│   ├── crop_recommendation/          # Scikit-Learn training, prediction & evaluation
│   ├── farmer_assistant/             # Gemini REST provider, LocalRule fallback & guardrails
│   ├── smart_irrigation/             # Deterministic 5-tier irrigation heuristics
│   ├── sustainability/               # Multi-dimensional sustainability scoring engine
│   └── weather_intelligence/         # Agrometeorological rules & caching logic
├── report/                           # Model reports & evaluation artifacts
│   ├── crop_recommendation/          # Crop recommendation metrics & confusion matrix
│   └── disease_model_report.md       # One-page Core Disease Detection model report
├── scripts/                          # Dataset verification and setup utilities
├── tests/                            # Comprehensive automated test suite (399 tests)
├── .env.example                      # Safe environment configuration template
├── .gitignore                        # Git exclusion rules protecting secrets & temporary data
├── requirements.txt                  # Python runtime dependencies
└── README.md                         # Official SIH project documentation
```

---

## 🔬 Implemented Modules

### 1. Core Module — Crop Disease Detection (Computer Vision)
- **Purpose**: Rapid foliar disease diagnosis from crop leaf photographs to assist farmers with early intervention and prevent epidemic crop loss.
- **Technology**: Deep Convolutional Transfer Learning using PyTorch and Torchvision. ImageNet-pretrained ResNet-50 backbone with a custom linear classification head.
- **Inference Interface**:
  - Python interface: `predict(image_path: str | Path, checkpoint_path=None, device="cpu", top_k=3) -> dict` in [`model/predict.py`](model/predict.py).
  - CLI: `python model/predict.py --image path/to/leaf.jpg --top-k 5`
- **Application Integration**: [`app/services/disease_service.py`](app/services/disease_service.py) automatically inspects whether a trained checkpoint is available. If a trained checkpoint is present, it executes live inference. If not present, it operates in an explicitly labeled Development Mock Mode with an amber banner to prevent false claims.
- **Primary Metrics**: Macro-averaged F1 score, Top-1 accuracy, multi-class confusion matrix, per-class precision and recall.
- **Official SIH Evaluation Status**:
  > **"Final held-out SIH field-test Macro-F1 is pending organizer-provided evaluation data. No fabricated score is reported."**
  >
  > *Integrity Policy*: The SIH problem statement stipulates that final evaluation must be conducted on an unseen, organizer-provided field-condition test set. In strict compliance with hackathon ethics, no synthetic metrics, fabricated confusion matrices, or baseline comparisons are reported as official SIH results.

### 2. Bonus Module A — Crop Recommendation (Tabular Machine Learning)
- **Purpose**: Recommends the most suitable and climate-resilient crop varieties based on soil chemical parameters and climatic factors.
- **Technology**: `sklearn.ensemble.RandomForestClassifier` (100 estimators, random state 42).
- **Dataset**: Public domain precision agriculture benchmark (2,200 samples across 22 balanced crops; 100 samples per crop).
- **Features Modeled (7 ML Features)**: Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), Temperature, Humidity, Soil pH, and Rainfall.
- **Data Split Methodology**: Stratified 70% Train (1,540 rows) / 15% Validation (330 rows) / 15% Held-out Test (330 rows) with zero index leakage.
- **Verified Held-Out Test Metrics**:
  - **Test Accuracy**: **99.39%** (328 / 330 correct classifications)
  - **Macro-F1 Score**: **0.9939**
  - **Weighted-F1 Score**: **0.9939**
  - Full 22x22 confusion matrix and classification report are serialized in [`report/crop_recommendation/`](report/crop_recommendation/).
- **Notice on Scope**: These verified metrics apply strictly and exclusively to the tabular Crop Recommendation module; they are not conflated with Disease Detection.

### 3. Bonus Module B — Smart Irrigation Advisor (Deterministic Rule-Based)
- **Purpose**: Provides actionable irrigation timing recommendations to prevent drought stress while avoiding over-watering and wasteful aquifer depletion.
- **Technology**: Deterministic, explainable agrometeorological expert heuristic engine (100% rule-based, NOT machine learning).
- **Inputs**: Farmer-reported soil moisture percentage (0–100%), 7-day precipitation forecast (probability and quantitative mm amount), crop type, and crop growth stage.
- **Decision Hierarchy**:
  - *Tier A (Emergency Deficit)*: Soil moisture $\le 25\%$; prioritizes urgent irrigation review unless heavy soaking rain ($\ge 10\text{ mm}$) is imminent within 24 hours.
  - *Tier B (Significant Near-Term Rain)*: Rainfall $\ge 8\text{ mm}$ or $\ge 60\%$ probability with $\ge 2\text{ mm}$; advises delaying irrigation.
  - *Tier C (Depleted Moisture)*: Soil moisture $\le 35\%$ with dry forecast; advises considering irrigation.
  - *Tier D (Transitional)*: 30–59% rain likelihood; advises monitoring without immediate irrigation.
  - *Tier E (Adequate Moisture)*: Soil moisture $\ge 65\%$; advises deferring irrigation.
- **Explainability**: Every recommendation outputs a clear "Why This Recommendation?" explanation showing the exact trigger criteria and evaluated inputs.

### 4. Bonus Module C — Weather-Based Intelligence (Agrometeorological Rules)
- **Purpose**: Real-time atmospheric monitoring and 7-day forward-looking risk assessment to assist farm scheduling.
- **Technology**: Live integration with [Open-Meteo APIs](https://open-meteo.com/en/docs) with 15-minute in-memory caching and manual on-demand refresh.
- **Features**:
  - 7-day temperature, precipitation sum, and rain probability forecasts.
  - Foliar disease microclimate risk index based on prolonged relative humidity ($\ge 75\%$).
  - Crop heat stress warnings ($\ge 35^\circ\text{C}$) tracking growth stage sensitivity.
  - Chemical spray drift windows based on wind speed ($\ge 20\text{ km/h}$) and active rainfall.
- **Operational Mode**: LIVE mode queries Open-Meteo. If network is unavailable or rate-limited, safely transitions to an explicitly badged DEMO mode with deterministic simulated data and a retry option.

### 5. Bonus Module D — Farm Sustainability Score (Deterministic Framework)
- **Purpose**: Evaluates overall on-farm environmental stewardship across water, soil, and crop health dimensions.
- **Technology**: Multi-dimensional deterministic formula with dynamic missing-data weight renormalization and FAO / ICAR crop water benchmarks.
- **Formula**:
  $$\text{Sustainability Score} = \frac{\sum_{i \in \text{Available}} (\text{Score}_i \times W_i)}{\sum_{i \in \text{Available}} W_i}$$
  - Water Efficiency ($W = 40\%$): Evaluates irrigation method baseline (Drip: 90, Sprinkler: 75, Flood: 50, Rainfed: 85), soil moisture balance, and weather opportunities.
  - Resource & Soil Conservation ($W = 30\%$): Evaluates nutrient practices (Organic: 95, Integrated: 85, Moderate: 70, Intensive Chemical: 45) and soil cover (Cover crops, mulching, conservation tillage).
  - Crop Foliar Health ($W = 30\%$): Evaluates foliar vigor (Healthy: 95, Stress: 75, Disease: 45, Severe: 25).
- **Dynamic Renormalization**: If foliar health is unassessed, its 30% weight is omitted from the denominator, renormalizing Water ($57.1\%$) and Soil ($42.9\%$) without fabricating data or penalizing the farmer.
- **Agronomic Benchmarks**: Integrates seasonal crop water requirements ($ET_c$) and critical stage depletion fractions from FAO Irrigation and Drainage Paper 56, Paper 33, and ICAR references ([`data/sustainability/crop_requirements.csv`](data/sustainability/crop_requirements.csv)).

### 6. Bonus Module E — Farmer Assistant (Grounded GenAI & Local Rule Fallback)
- **Purpose**: Conversational advisory assistant answering farmer queries in plain language (English, Hindi, Gujarati).
- **Technology**: Multi-provider architecture supporting Google Gemini Free Tier (`gemini-flash-lite-latest`) via REST (`:generateContent`) with automatic fallback to deterministic `LocalRuleProvider`.
- **Grounding**: Grounded strictly in active AgriSmart module results (Disease diagnosis, Crop suitability, Weather conditions, Irrigation timing, and Sustainability scores). Missing values are explicitly flagged as `UNAVAILABLE` or `NOT_PROVIDED`.
- **Security & Safety**: Server-side API key management (never exposed to client), prompt-injection defenses against system prompt or environment variable extraction, and refusal guardrails for hazardous chemical synthesis or off-label pesticide recommendations.
- **Session Memory**: Compact session storage strictly isolating farm context parameters (`session["farm_context"]`) without storing raw API keys, chat transcripts, or large model payloads in cookies.

---

## 🚫 Modules Intentionally Not Implemented

- **IoT Integration (Module F)**: **Intentionally Excluded**. AgriSmart AI is strictly designed as a zero-hardware, zero-sensor solution. All soil moisture and field parameters are entered manually by the farmer based on physical observations. No fake IoT telemetry, hardware simulators, or microcontrollers are used.
- **Agentic Advisor (Module G)**: **Not Implemented**. Autonomous multi-step farm intervention planning is an un-implemented future research stub ([`app/services/advisor_service.py`](app/services/advisor_service.py)) and is not claimed as a completed feature.

---

## 📊 Datasets Used & Attributions

| Dataset | Module | Source / Reference | License | Role in Project |
|---|---|---|---|---|
| **PlantVillage** | Disease Detection | Hughes & Salathé (2015), Penn State / EPFL ([GitHub Repo](https://github.com/spMohanty/PlantVillage-Dataset)) | CC0: Public Domain | Training and validation pipeline foundation for foliar disease classification. |
| **Official SIH Held-out Field Dataset** | Disease Detection | Smart India Hackathon 2026 Organizers | Proprietary / SIH Organizers | Official unseen test set for hackathon evaluation. **Must NEVER be used for training.** Status: *Pending organizer release.* |
| **Crop Recommendation Benchmark** | Crop Recommendation | Atharva Inamdar / gabbygab1233 ([GitHub](https://raw.githubusercontent.com/gabbygab1233/Crop-Recommender/main/Crop_recommendation.csv)) | CC0: Public Domain | Tabular dataset (2,200 rows, 22 crops, 7 features) used for Random Forest training and held-out validation. |
| **Crop Water Requirements ($ET_c$) & Depletion ($p$)** | Sustainability Score | United Nations FAO Irrigation & Drainage Paper 56, Paper 33, and ICAR Water Management | Open Academic / UN FAO | Empirical reference data for 17 major crops ([`data/sustainability/crop_requirements.csv`](data/sustainability/crop_requirements.csv)). |

---

## ⚡ Reproducibility & Run Instructions

All commands use relative repository paths and work on Linux, macOS, and Windows.

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgriSmart-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for GenAI)

Copy the safe example environment template:
```bash
cp .env.example .env
```
To enable Google Gemini for the Farmer Assistant, set your key in `.env`:
```env
FARMER_ASSISTANT_PROVIDER=gemini
FARMER_ASSISTANT_API_KEY=your_gemini_api_key_here
FARMER_ASSISTANT_MODEL=gemini-flash-lite-latest
FARMER_ASSISTANT_TIMEOUT=10
```
*Note: If no API key is provided, the Farmer Assistant automatically runs in **Assistant Demo · Local Rule Mode** with zero external dependencies.*

### 3. Start the Web Application

```bash
python -m app.main
```
Open your browser and navigate to: **`http://127.0.0.1:5050`**

### 4. Run the Automated Test Suite

```bash
pytest -v
```
All tests should pass with 100% success rate.

### 5. Run Disease Detection Prediction (CLI)

```bash
# Inference using documented CLI interface
python model/predict.py --image path/to/leaf_image.jpg --top-k 5

# Programmatic Python usage:
# from model.predict import predict
# result = predict("path/to/leaf_image.jpg")
# print(result["predicted_class"], result["confidence"])
```

### 6. Run Crop Recommendation Prediction (CLI)

```bash
python -m model.crop_recommendation.predict --ph 6.5 --temp 28.0 --humidity 70.0 --rainfall 120.0 --n 50 --p 50 --k 50
```

---

## 📄 One-Page Disease Model Report

A dedicated one-page report for the Core Crop Disease Detection model is available at:
**[`report/disease_model_report.md`](report/disease_model_report.md)**

---

## 🔒 Security & Privacy Audit

- **Zero Committed Secrets**: `.env` and `.env.*` are strictly excluded in `.gitignore`. `.env.example` contains only placeholder values.
- **Server-Side API Handling**: API keys are accessed exclusively server-side via Python environment variables. No keys or tokens are rendered in HTML templates or sent to client-side JavaScript.
- **No Hardcoded Machine Paths**: All internal paths use repository-relative `pathlib.Path` structures without machine-specific usernames or drive letters.
- **Sanitized Uploads**: Uploaded leaf images are processed in a dedicated directory ignored by Git, with strict MIME-type and size validation (16 MB limit).

---

## 📝 Originality & Third-Party Attribution

In compliance with SIH 2026 hackathon regulations:
- **AI Assistance**: AI coding assistants (Google Antigravity / Gemini) were utilized for code structuring, testing, and documentation assistance during project development.
- **Open-Source Libraries**: AgriSmart AI builds upon established open-source libraries: PyTorch, Torchvision, Scikit-Learn, Flask, NumPy, Pandas, Pillow, OpenCV, Requests, and Pytest.
- **Public Datasets**: Public datasets (PlantVillage for computer vision; precision agriculture crop recommendation dataset by Atharva Inamdar / gabbygab1233; FAO 56/33 agronomic tables) are credited and cited.
- **Original Architecture**: All service boundary implementations, agrometeorological heuristic rules, dynamic sustainability weight renormalization formulas, grounded conversational fallbacks, and user interfaces are original implementations developed for this project. No public repository or competition notebook was copied wholesale.



- **License**: MIT License / Open Source
