"""
POST /api/v1/score

Request  → list of member records (Pydantic-validated)
Response → { success, data: [...], timestamp, run_id }

Each data item: { hcid, is_anomaly, score, severity, anomaly_reasons }

The Scorer singleton is loaded lazily on first request.
If the model artifacts are absent (e.g. first boot before training),
the endpoint returns HTTP 503 with an informative message.

Guardrails
----------
* Empty payload                 → HTTP 422
* Records missing required keys → HTTP 422 with field-level errors
* Model not loaded              → HTTP 503
* Unexpected scorer error       → HTTP 500 (detail hidden in production)
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, field_validator, model_validator

from app.config.settings import get_settings
from app.core.exceptions import (
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
    ModelNotTrainedError,
)
from app.core.logging import get_logger, set_run_id
from app.model.scorer import Scorer, ScoredRecord
from app.model.severity import Severity

logger = get_logger(__name__)
router = APIRouter(tags=["anomaly-detection"])


# ── request / response schemas ────────────────────────────────────────────────


class MemberRecord(BaseModel):
    """
    A single member enrollment record.

    All fields are optional to support partial records from upstream systems.
    The scorer handles missing values gracefully — they contribute to the
    MISSING_* anomaly flags rather than causing errors.

    Column names follow the CSV header exactly; aliases accommodate both the
    space-separated original ("CONT EFF") and underscore form ("CONT_EFF").
    """
    HCID:     Optional[str]   = Field(None, description="Member health-care ID")
    SSN:      Optional[str]   = Field(None, description="Social Security Number")
    CERT:     Optional[str]   = None
    MCNT:     Optional[int]   = Field(None, ge=1, le=100, description="Member count (family size)")
    MCDE:     Optional[int]   = Field(None, ge=10, le=79, description="Member role code")
    MEMBIRDT: Optional[int]   = Field(None, description="Member birth date YYYYMMDD")
    CONT_EFF: Optional[int]   = Field(None, alias="CONT EFF", description="Contract effective date YYYYMMDD")
    CONT_CAN: Optional[int]   = Field(None, alias="CONT CAN", description="Contract cancel date YYYYMMDD (0=active)")
    MEMEFFDT: Optional[int]   = None
    MEMCANDT: Optional[int]   = None
    STS:      Optional[int]   = Field(None, ge=0, le=4)
    CR:       Optional[str]   = None
    MBUTY:    Optional[str]   = Field(None, pattern=r"^(IND|SMGRP|SEN)$")
    TYP:      Optional[str]   = Field(None, pattern=r"^(MED|DEN|VIS|LFE|STD|LTD)$")
    TP1:      Optional[str]   = Field(None, pattern=r"^(PPO|HMO|POS|EPO|VIS)?$")
    ST:       Optional[str]   = Field(None, max_length=2)
    ETHNI:    Optional[str]   = None
    GROUP:    Optional[str]   = None
    BROKER:   Optional[str]   = None
    PRVDR:    Optional[str]   = None
    PRE:      Optional[float] = Field(None, ge=0.0)
    CANPRODT: Optional[int]   = None
    CONT:     Optional[str]   = None
    CNTDT:    Optional[int]   = None
    PDLN:     Optional[str]   = None
    TP2:      Optional[str]   = None
    XI:       Optional[str]   = None

    model_config = {"populate_by_name": True}

    @field_validator("MEMBIRDT")
    @classmethod
    def _validate_birthdate(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not (18000101 <= v <= 20991231):
            raise ValueError(f"MEMBIRDT {v} is not a valid YYYYMMDD date")
        return v

    @field_validator("MCDE")
    @classmethod
    def _validate_mcde(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        valid = {10, 20, 30, 40, 50, 60, 70}
        if v not in valid and not (10 <= v <= 79):
            raise ValueError(f"MCDE {v} is not a recognised member code")
        return v


_SCORE_REQUEST_EXAMPLE = {
    "records": [
        {
            "HCID": "235A06140",
            "SSN": "",
            "MCNT": 1,
            "MCDE": 20,
            "MEMBIRDT": 19520728,
            "CONT EFF": 20060501,
            "CONT CAN": 20060501,
            "MEMEFFDT": 20060501,
            "MEMCANDT": 20060501,
            "STS": 1,
            "CR": "N",
            "MBUTY": "SMGRP",
            "TYP": "LFE",
            "TP1": "PPO",
            "ST": "CO",
            "PRE": 150.0,
            "BROKER": "Y",
            "PRVDR": "Y"
        }
    ]
}


class ScoreRequest(BaseModel):
    records: list[MemberRecord] = Field(..., min_length=1, max_length=5000)

    model_config = {
        "json_schema_extra": {"example": _SCORE_REQUEST_EXAMPLE}
    }

    @model_validator(mode="before")
    @classmethod
    def _reject_empty(cls, values: Any) -> Any:
        records = values.get("records") if isinstance(values, dict) else None
        if isinstance(records, list) and len(records) == 0:
            raise ValueError("records list must not be empty")
        return values


class ScoredRecordOut(BaseModel):
    hcid:            Optional[str]
    is_anomaly:      bool
    score:           float          # [0, 1]
    raw_score:       float
    severity:        Severity
    anomaly_reasons: list[str]


class ScoreResponse(BaseModel):
    success:       bool
    run_id:        str
    timestamp:     str
    total_records: int
    anomaly_count: int
    normal_count:  int
    anomaly_rate:  float
    data:          list[ScoredRecordOut]


# ── scorer singleton ──────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_scorer() -> Scorer:
    """Load the Scorer singleton once and cache it for the lifetime of the process."""
    settings = get_settings()
    scorer   = Scorer()
    scorer.load(settings.MODEL_DIR)
    return scorer


# ── endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score member records for enrollment anomalies",
    description=(
        "Accepts 1–5000 member enrollment records and returns a normalised "
        "anomaly score [0,1], a CRITICAL/HIGH/MEDIUM/LOW severity label, and "
        "rule-based anomaly reasons for each record."
    ),
)
async def score_records(payload: ScoreRequest, request: Request) -> ScoreResponse:
    run_id = str(uuid.uuid4())
    set_run_id(run_id)

    logger.info(
        "score_records called",
        extra={"run_id": run_id, "records": len(payload.records)},
    )

    # ── load model (raises 503 if artifacts missing) ─────────────────────────
    try:
        scorer = _get_scorer()
    except (ModelNotFoundError, ModelLoadError) as exc:
        logger.error("Model unavailable", extra={"run_id": run_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Anomaly detection model is not available. "
                "Run the training script to generate model artifacts first."
            ),
        ) from exc

    # ── convert pydantic records → DataFrame ─────────────────────────────────
    rows = []
    for rec in payload.records:
        d = rec.model_dump(by_alias=True)
        rows.append(d)
    df = pd.DataFrame(rows)

    # ── score ────────────────────────────────────────────────────────────────
    try:
        results: list[ScoredRecord] = scorer.score(df)
    except ModelNotTrainedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        settings = get_settings()
        detail   = str(exc) if settings.ENV != "production" else "Internal scoring error"
        logger.error("Scoring error", extra={"run_id": run_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from exc

    # ── build response ────────────────────────────────────────────────────────
    out_records = [
        ScoredRecordOut(
            hcid            = r.hcid,
            is_anomaly      = r.is_anomaly,
            score           = r.score,
            raw_score       = r.raw_score,
            severity        = r.severity,
            anomaly_reasons = r.anomaly_reasons,
        )
        for r in results
    ]

    anomaly_count = sum(1 for r in results if r.is_anomaly)
    return ScoreResponse(
        success       = True,
        run_id        = run_id,
        timestamp     = datetime.now(timezone.utc).isoformat(),
        total_records = len(results),
        anomaly_count = anomaly_count,
        normal_count  = len(results) - anomaly_count,
        anomaly_rate  = round(anomaly_count / max(len(results), 1), 4),
        data          = out_records,
    )


# ── CSV upload endpoint ───────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
}
_MAX_CSV_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/score/csv",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score an entire CSV file for enrollment anomalies",
    description=(
        "Upload a CSV file (up to 50 MB) matching the insurance enrollment format. "
        "Every row is scored. By default only anomalies are returned in `data` to keep "
        "the response small. Set `anomalies_only=false` to include all records."
    ),
)
async def score_csv(
    request: Request,
    file: UploadFile = File(..., description="CSV file with insurance enrollment records"),
    anomalies_only: bool = True,
) -> ScoreResponse:
    run_id = str(uuid.uuid4())
    set_run_id(run_id)

    # ── validate content type ─────────────────────────────────────────────────
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    filename = file.filename or ""
    if content_type not in _ALLOWED_CONTENT_TYPES and not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a CSV file, received content-type '{file.content_type}'.",
        )

    # ── read and size-check ───────────────────────────────────────────────────
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(raw_bytes) > _MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the 50 MB limit ({len(raw_bytes) / 1_048_576:.1f} MB received).",
        )

    # ── parse CSV ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file contains no data rows.",
        )

    logger.info(
        "score_csv called",
        extra={"run_id": run_id, "csv_filename": filename, "rows": len(df), "cols": len(df.columns)},
    )

    # ── drop known empty/irrelevant columns if present ────────────────────────
    df.drop(columns=[c for c in ("RX", "Unnamed: 37") if c in df.columns], inplace=True)

    # ── load model ────────────────────────────────────────────────────────────
    try:
        scorer = _get_scorer()
    except (ModelNotFoundError, ModelLoadError) as exc:
        logger.error("Model unavailable", extra={"run_id": run_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Anomaly detection model is not available. "
                "Run the training script to generate model artifacts first."
            ),
        ) from exc

    # ── score ────────────────────────────────────────────────────────────────
    try:
        results: list[ScoredRecord] = scorer.score(df)
    except ModelNotTrainedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        settings = get_settings()
        detail   = str(exc) if settings.ENV != "production" else "Internal scoring error"
        logger.error("CSV scoring error", extra={"run_id": run_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from exc

    # ── build response ────────────────────────────────────────────────────────
    anomaly_count = sum(1 for r in results if r.is_anomaly)

    filtered = [r for r in results if r.is_anomaly] if anomalies_only else results
    out_records = [
        ScoredRecordOut(
            hcid            = r.hcid,
            is_anomaly      = r.is_anomaly,
            score           = r.score,
            raw_score       = r.raw_score,
            severity        = r.severity,
            anomaly_reasons = r.anomaly_reasons,
        )
        for r in filtered
    ]

    return ScoreResponse(
        success       = True,
        run_id        = run_id,
        timestamp     = datetime.now(timezone.utc).isoformat(),
        total_records = len(results),
        anomaly_count = anomaly_count,
        normal_count  = len(results) - anomaly_count,
        anomaly_rate  = round(anomaly_count / max(len(results), 1), 4),
        data          = out_records,
    )
