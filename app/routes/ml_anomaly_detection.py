"""
FastAPI Routes for ML Anomaly Detection
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any
import logging

from ml.inference import get_anomaly_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml/anomaly-detection", tags=["ML Anomaly Detection"])


@router.get("/detect")
async def detect_anomalies(
    model: str = Query(
        "isolation_forest",
        description="Model to use: isolation_forest, local_outlier_factor, one_class_svm, elliptic_envelope"
    )
) -> Dict[str, Any]:
    """
    Detect anomalies using a specific ML model
    """
    try:
        service = get_anomaly_service()
        result = service.detect_anomalies(model_name=model)
        return result
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detect-ensemble")
async def detect_anomalies_ensemble(
    threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Ensemble threshold (0.5 = majority vote)"
    )
) -> Dict[str, Any]:
    """
    Detect anomalies using ensemble of all models
    """
    try:
        service = get_anomaly_service()
        result = service.detect_anomalies_ensemble(threshold=threshold)
        return result
    except Exception as e:
        logger.error(f"Ensemble anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/member/{hcid}")
async def get_member_anomaly_details(hcid: str) -> Dict[str, Any]:
    """
    Get anomaly detection details for a specific member
    """
    try:
        service = get_anomaly_service()
        result = service.get_anomaly_details(hcid)
        
        if not result.get('success'):
            raise HTTPException(status_code=404, detail=result.get('error'))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get member details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_anomaly_statistics() -> Dict[str, Any]:
    """
    Get overall anomaly detection statistics
    """
    try:
        service = get_anomaly_service()
        result = service.get_anomaly_statistics()
        return result
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """
    List all available anomaly detection models
    """
    return {
        "success": True,
        "models": [
            {
                "name": "isolation_forest",
                "description": "Isolation Forest - Best for high-dimensional data, fast training",
                "type": "Tree-based ensemble"
            },
            {
                "name": "local_outlier_factor",
                "description": "Local Outlier Factor - Best for density-based anomalies",
                "type": "Density-based"
            },
            {
                "name": "one_class_svm",
                "description": "One-Class SVM - Best for non-linear boundaries",
                "type": "Support Vector Machine"
            },
            {
                "name": "elliptic_envelope",
                "description": "Elliptic Envelope - Best for Gaussian distributed data",
                "type": "Covariance-based"
            }
        ],
        "ensemble": {
            "description": "Ensemble combines all models for robust detection",
            "recommendation": "Use ensemble for production with threshold=0.5"
        }
    }
