from fastapi import APIRouter, Query
from typing import Dict, Any
from app.services.enrollment_insights_service import EnrollmentInsightsService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/enrollment", tags=["Enrollment Insights"])

enrollment_service = EnrollmentInsightsService()


@router.get("/summary")
async def get_enrollment_summary() -> Dict[str, Any]:
    result = await enrollment_service.get_enrollment_summary()
    return result


@router.get("/by-state")
async def get_enrollment_by_state() -> Dict[str, Any]:
    result = await enrollment_service.get_enrollment_by_state()
    return result


@router.get("/trends")
async def get_enrollment_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
) -> Dict[str, Any]:
    result = await enrollment_service.get_enrollment_trends(days=days)
    return result


@router.get("/validation-errors")
async def get_validation_errors(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of errors to return")
) -> Dict[str, Any]:
    result = await enrollment_service.get_validation_errors(limit=limit)
    return result


@router.get("/maintenance-types")
async def get_maintenance_type_distribution() -> Dict[str, Any]:
    result = await enrollment_service.get_maintenance_type_distribution()
    return result


@router.get("/file-processing")
async def get_file_processing_stats() -> Dict[str, Any]:
    result = await enrollment_service.get_file_processing_stats()
    return result
