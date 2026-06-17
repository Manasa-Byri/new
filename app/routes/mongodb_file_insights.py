from fastapi import APIRouter, Query
from typing import Dict, Any
from app.services.mongodb_file_insights_service import MongoDBFileInsightsService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/mongodb/files", tags=["MongoDB File Insights"])

file_service = MongoDBFileInsightsService()


@router.get("/summary")
async def get_file_ingestion_summary() -> Dict[str, Any]:
    result = await file_service.get_file_ingestion_summary()
    return result


@router.get("/by-status")
async def get_files_by_status() -> Dict[str, Any]:
    result = await file_service.get_files_by_status()
    return result


@router.get("/by-source-system")
async def get_files_by_source_system() -> Dict[str, Any]:
    result = await file_service.get_files_by_source_system()
    return result


@router.get("/format-distribution")
async def get_file_format_distribution() -> Dict[str, Any]:
    result = await file_service.get_file_format_distribution()
    return result


@router.get("/processing-performance")
async def get_processing_performance() -> Dict[str, Any]:
    result = await file_service.get_processing_performance()
    return result


@router.get("/recent-uploads")
async def get_recent_uploads(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back")
) -> Dict[str, Any]:
    result = await file_service.get_recent_uploads(hours=hours)
    return result


@router.get("/error-analysis")
async def get_error_analysis() -> Dict[str, Any]:
    result = await file_service.get_error_analysis()
    return result
