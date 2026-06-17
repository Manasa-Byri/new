"""
POST /api/v1/predict/anomaly
Accepts raw member data, runs the trained IsolationForest pipeline,
returns anomaly scores and status per record.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Anomaly Prediction"])


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT SCHEMA  (mirrors raw CSV columns — all optional for flexibility)
# ═══════════════════════════════════════════════════════════════════════════════

class MemberRecord(BaseModel):
    HCID:      Optional[str]   = Field(None,  description="Health Care ID")
    SSN:       Optional[float] = Field(None,  description="Social Security Number")
    CERT:      Optional[str]   = Field(None,  description="Certificate Number")
    MCNT:      Optional[int]   = Field(1,     description="Member count in contract")
    MCDE:      Optional[int]   = Field(10,    description="Member code (10/20=Primary, 30-70=Dependent)")
    MEMBIRDT:  Optional[int]   = Field(0,     description="Date of birth YYYYMMDD")
    CONT_EFF:  Optional[int]   = Field(0,     description="Contract effective date YYYYMMDD", alias="CONT EFF")
    CONT_CAN:  Optional[int]   = Field(0,     description="Contract cancel date YYYYMMDD (0=not cancelled)", alias="CONT CAN")
    MEMEFFDT:  Optional[int]   = Field(0,     description="Member effective date YYYYMMDD")
    MEMCANDT:  Optional[int]   = Field(0,     description="Member cancel date YYYYMMDD (0=not cancelled)")
    STS:       Optional[int]   = Field(0,     description="Status (0=Active, 1=Inactive, 2=Suspended, 4=Terminated)")
    CR:        Optional[Any]   = Field(None,  description="Cancellation reason code")
    MBUTY:     Optional[str]   = Field("IND", description="Business type (IND, SMGRP, SEN)")
    TYP:       Optional[str]   = Field("MED", description="Coverage type (MED, DEN, VIS, LFE, STD, LTD)")
    TP1:       Optional[str]   = Field("PPO", description="Plan type (PPO, HMO, POS, EPO)")
    ST:        Optional[str]   = Field("NY",  description="State code")
    ETHNI:     Optional[str]   = Field("99",  description="Ethnicity code")

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    records: list[MemberRecord] = Field(..., description="One or more member records to score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "HCID": "172T97147",
                        "SSN": None,
                        "CERT": "AAA000066",
                        "MCNT": 1,
                        "MCDE": 10,
                        "MEMBIRDT": 19900201,
                        "CONT EFF": 20160101,
                        "CONT CAN": 0,
                        "MEMEFFDT": 20160101,
                        "MEMCANDT": 0,
                        "STS": 0,
                        "CR": None,
                        "MBUTY": "IND",
                        "TYP": "MED",
                        "TP1": "HMO",
                        "ST": "NY",
                        "ETHNI": "AMERC"
                    }
                ]
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    hcid:             str
    is_anomaly:       bool
    anomaly_status:   str   = Field(description="'ANOMALY' or 'NORMAL'")
    anomaly_score:    float = Field(description="0–1; higher = more anomalous")
    raw_score:        float = Field(description="Raw IsolationForest decision function value")
    anomaly_reasons:  list[str]


class PredictResponse(BaseModel):
    success:        bool
    total_records:  int
    anomaly_count:  int
    normal_count:   int
    anomaly_rate:   float = Field(description="Percentage of anomalies in this batch")
    predictions:    list[PredictionResult]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL SINGLETON  (loaded once, reused for every request)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_pipeline():
    """Load trained pipeline from disk.  Cached after first call."""
    from ml.insurance_anomaly import AnomalyPipeline, MODEL_PATH, PREPROCESSOR_PATH
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: "
            "`.\\venv\\Scripts\\python.exe ml/train_insurance.py` first."
        )
    logger.info("Loading insurance anomaly pipeline from disk …")
    return AnomalyPipeline.load()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/anomaly", response_model=PredictResponse, summary="Predict anomalies for member records")
async def predict_anomaly(request: PredictRequest):
    """
    **POST** one or more insurance member records and receive back:

    - `is_anomaly` — `true` / `false`
    - `anomaly_status` — `"ANOMALY"` or `"NORMAL"`
    - `anomaly_score` — 0 to 1 (higher = more suspicious)
    - `raw_score` — raw IsolationForest decision function value
    - `anomaly_reasons` — human-readable list of why the record is flagged

    All fields map directly to CSV columns.
    Use `CONT EFF` / `CONT CAN` as JSON keys (with space), or the aliases
    `CONT_EFF` / `CONT_CAN` (with underscore) — both are accepted.
    """
    if not request.records:
        raise HTTPException(status_code=422, detail="records list must not be empty")

    try:
        pipeline = _get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── convert Pydantic models → DataFrame ───────────────────────────────────
    rows = []
    for rec in request.records:
        d = rec.model_dump(by_alias=True)
        # normalise alias keys back to original column names
        if "CONT_EFF" in d:
            d["CONT EFF"] = d.pop("CONT_EFF")
        if "CONT_CAN" in d:
            d["CONT CAN"] = d.pop("CONT_CAN")
        rows.append(d)

    df = pd.DataFrame(rows)

    # fill columns required by preprocessor but absent from payload
    _defaults = {
        "CONT EFF": 0, "CONT CAN": 0, "MEMEFFDT": 0, "MEMCANDT": 0,
        "MEMBIRDT": 0, "STS": 0, "MCNT": 1, "MCDE": 10,
        "MBUTY": "IND", "TYP": "MED", "TP1": "PPO", "ST": "NY",
        "ETHNI": "99", "CR": None,
    }
    for col, default in _defaults.items():
        if col not in df.columns:
            df[col] = default

    # ── predict ───────────────────────────────────────────────────────────────
    try:
        raw_predictions = pipeline.predict_df(df)
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    predictions = [PredictionResult(**p) for p in raw_predictions]
    anomaly_count = sum(1 for p in predictions if p.is_anomaly)

    return PredictResponse(
        success=True,
        total_records=len(predictions),
        anomaly_count=anomaly_count,
        normal_count=len(predictions) - anomaly_count,
        anomaly_rate=round(anomaly_count / len(predictions) * 100, 2),
        predictions=predictions,
    )
