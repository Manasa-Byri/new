# AI Insight Agent — Anomaly Detection Service v1

A **production-ready, read-only anomaly detection micro-service** for a health-insurance enrollment platform.

The service continuously observes member enrollment data from PostgreSQL, MongoDB, and CSV sources, scores each record with an Isolation Forest model, and returns a normalised anomaly score, a severity label (CRITICAL / HIGH / MEDIUM / LOW), and human-readable anomaly reasons.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
  HTTP client  ───► │  POST /api/v1/score                  │
                    │  FastAPI  (app/api/routes.py)         │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  Scorer  (app/model/scorer.py)        │
                    │  ├── build_features()                 │
                    │  ├── IsolationForest.predict()        │
                    │  ├── classify_severity()              │
                    │  └── _explain_batch()  (15 vectorised rules) │
                    └──────────────────────────────────────┘
```

**Model choice — Isolation Forest:**
- Fully unsupervised (no labelled fraud examples required)
- Handles mixed tabular data natively; trains in seconds on 50k rows
- Reproducible: `random_state=42` baked into `Settings`
- Rejected: Autoencoder (GPU, opacity), One-Class SVM (O(n²) complexity)

---

## Project Structure

```
app/
  api/
    routes.py                       — POST /score and POST /score/csv (main ML scoring)
  config/
    settings.py                     — env-var driven config (no hardcoded secrets)
  core/
    logging.py                      — structured JSON logging + run-id context
    exceptions.py                   — typed domain exceptions
  data/
    loaders.py                      — read-only CSV / PostgreSQL / MongoDB loaders
  middleware/
    error_handler.py                — centralised error → structured JSON response
    logging_middleware.py           — request/response logging for every call
    rate_limiter.py                 — per-IP rate limiting (configurable)
  model/
    features.py                     — build_features(): 36 columns → 28 float32 features
    scorer.py                       — Scorer: fit / save / load / score + _explain_batch()
    severity.py                     — classify_severity(): score → CRITICAL/HIGH/MEDIUM/LOW
  models/
    db_models.py                    — SQLAlchemy table definitions
    schemas.py                      — Pydantic DB schemas
  routes/
    audit_insights.py               — GET /api/v1/insights/audit/*
    csv_insurance_insights.py       — GET /api/v1/insights/csv/*  (15 endpoints)
    datasources.py                  — GET /api/v1/datasources/*
    enrollment_insights.py          — GET /api/v1/insights/enrollment/*
    health.py                       — GET /api/v1/health
    insights.py                     — legacy insights routing
    ml_anomaly_detection.py         — GET /api/v1/ml/anomaly-detection/*
    mongodb_file_insights.py        — GET /api/v1/insights/mongodb/files/*
    mongodb_processing_insights.py  — GET /api/v1/insights/mongodb/processing/*
    predict_anomaly.py              — POST /api/v1/predict/anomaly (legacy)
    system_insights.py              — GET /api/v1/insights/system/*
  services/
    audit_insights_service.py
    base_service.py                 — base service interface
    cloudwatch_service.py           — AWS CloudWatch integration
    csv_insurance_insights_service.py
    database_service.py             — PostgreSQL query service
    enrollment_insights_service.py
    insight_aggregator.py           — service orchestrator
    mongodb_file_insights_service.py
    mongodb_processing_insights_service.py
    system_insights_service.py
    third_party_service.py          — third-party API calls
  app.py                            — FastAPI app factory + middleware + router registration
  constants.py                      — application-wide enums and constants
  database.py                       — SQLAlchemy engine + session factory

ml/
  models/
    v1_isolation_forest.joblib      — trained IsolationForest (200 trees)
    v1_preprocessor.joblib          — fitted StandardScaler (28 features)
  train.py                          — original multi-model training script
  train_insurance.py                — insurance-specific training pipeline
  anomaly_detector.py               — multi-model detector class
  inference.py                      — batch inference utilities
  insurance_anomaly.py              — insurance anomaly model wrapper
  data_preprocessing.py             — data preprocessing utilities
  config.py                         — ML hyperparameter configuration

specs/
  Technical_spec.md                 — full technical specification
  functional_spec.md                — functional specification

tests/
  test_features.py                  — feature engineering unit tests
  test_severity.py                  — severity threshold boundary tests
  test_scorer.py                    — Scorer unit tests
  test_loaders.py                   — CSV loader tests
  test_routes.py                    — API integration tests

train_v1.py          — standalone v1 model training entry point
run.py               — server entry point (uvicorn, port 8001)
Dockerfile
.env.example
requirements.txt
```

---

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials
```

### 2. Train

```bash
python train_v1.py
# Saves: ml/models/v1_isolation_forest.joblib
#        ml/models/v1_preprocessor.joblib
```

### 3. Run

```bash
python run.py
# http://localhost:8001/docs
```

### 4. Score records

```bash
curl -X POST http://localhost:8001/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"records": [{"HCID":"ABC123","MCNT":2,"MCDE":20,"MEMBIRDT":19850601,"CONT EFF":20200101,"CONT CAN":0,"STS":0,"MBUTY":"SMGRP","TYP":"MED","TP1":"PPO","ST":"TX"}]}'
```

**Response:**
```json
{
  "success": true,
  "run_id": "a1b2c3d4-...",
  "timestamp": "2026-01-01T12:00:00+00:00",
  "total_records": 1,
  "anomaly_count": 0,
  "normal_count": 1,
  "anomaly_rate": 0.0,
  "data": [{"hcid":"ABC123","is_anomaly":false,"score":0.12,"raw_score":-0.05,"severity":"LOW","anomaly_reasons":[]}]
}
```

### 5. Test

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
# Expected: 87 passed, 0 failed
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENV` | `development` | `development`/`staging`/`production` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MODEL_DIR` | `ml/models/` | Path to .joblib artifacts |
| `CONTAMINATION` | `0.05` | Expected anomaly fraction |
| `SEVERITY_CRITICAL_THRESHOLD` | `0.80` | Score ≥ → CRITICAL |
| `SEVERITY_HIGH_THRESHOLD` | `0.60` | Score ≥ → HIGH |
| `SEVERITY_MEDIUM_THRESHOLD` | `0.40` | Score ≥ → MEDIUM |
| `POSTGRES_HOST` | — | PostgreSQL host (skipped if absent) |
| `POSTGRES_DB` | — | Database name |
| `POSTGRES_USER` | — | Username |
| `POSTGRES_PASSWORD` | — | Password |
| `MONGODB_URL` | — | MongoDB connection string |
| `CSV_PATH` | auto-detect | Path to member CSV |

---

## Docker

```bash
docker build -t ai-insight-agent:v1 .
docker run -p 8001:8001 --env-file .env \
  -v $(pwd)/ml/models:/app/ml/models:ro \
  ai-insight-agent:v1
```

---

## Severity Levels

| Level | Score | Action |
|---|---|---|
| CRITICAL | ≥ 0.80 | Automated alert; immediate review |
| HIGH | ≥ 0.60 | Review within 24 hours |
| MEDIUM | ≥ 0.40 | Periodic batch review |
| LOW | < 0.40 | Normal; no action |

---

## Anomaly Flags

| Flag | Condition |
|---|---|
| `MISSING_SSN` | SSN null or empty |
| `OVERAGE_DEPENDENT` | Dependent age > 26 (TODO: confirm threshold with SME) |
| `MINOR_PRIMARY_SUBSCRIBER` | Primary subscriber age < 18 |
| `IND_MULTI_MEMBER` | Individual plan with MCNT > 1 |
| `IMMEDIATE_CANCEL` | Cancelled ≤ 7 days after start |
| `CANCEL_DATE_MISMATCH` | Contract/member cancel dates differ > 30 days |
| `ACTIVE_WITH_CANCEL_DATE` | Status=Active but has a cancel date |
| `UNUSUALLY_LARGE_FAMILY` | MCNT > 15 |
| `SENIOR_NON_SENIOR_PLAN` | Age ≥ 65 enrolled on non-Senior plan |
| `FUTURE_EFFECTIVE_DATE` | Contract effective date is in the future |
| `RETROACTIVE_CANCEL` | Cancel date is before the effective date |
| `ZERO_PREMIUM_ACTIVE` | Status=Active with premium = 0 |
| `MISSING_BIRTH_DATE` | Birth date is missing (MEMBIRDT = 0) |
| `MEMBER_CANCEL_WITHOUT_CONTRACT_CANCEL` | Member cancel date set but no contract cancel date |
| `MEMBER_EFFECTIVE_BEFORE_CONTRACT` | Member effective date is before contract effective date |

---

## Open TODOs (require SME sign-off)

1. Dependent age threshold: ACA law = **26**, internal policy may differ
2. AWD drop alert threshold: >5% confirmed or TBD?
3. Full `canonical_enrollments` column list — only 5 columns known
4. MongoDB connection string for production
5. `CONTAMINATION` prior — is 5% correct for this dataset?
