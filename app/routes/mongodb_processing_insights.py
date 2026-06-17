from fastapi import APIRouter, Query
from typing import Dict, Any
from app.services.mongodb_processing_insights_service import MongoDBProcessingInsightsService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/mongodb/processing", tags=["MongoDB Processing Insights"])

processing_service = MongoDBProcessingInsightsService()


@router.get("/summary")
async def get_processing_summary() -> Dict[str, Any]:
    result = await processing_service.get_processing_summary()
    return result


@router.get("/chunk-stats")
async def get_chunk_processing_stats() -> Dict[str, Any]:
    result = await processing_service.get_chunk_processing_stats()
    return result


@router.get("/member-status")
async def get_member_processing_by_status() -> Dict[str, Any]:
    result = await processing_service.get_member_processing_by_status()
    return result


@router.get("/audit-timeline")
async def get_processing_audit_timeline(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze")
) -> Dict[str, Any]:
    result = await processing_service.get_processing_audit_timeline(hours=hours)
    return result


@router.get("/recent-records")
async def get_recent_member_records(
    limit: int = Query(20, ge=1, le=100, description="Number of records to return")
) -> Dict[str, Any]:
    result = await processing_service.get_recent_member_records(limit=limit)
    return result


@router.get("/errors")
async def get_processing_errors(
    limit: int = Query(20, ge=1, le=100, description="Number of errors to return")
) -> Dict[str, Any]:
    result = await processing_service.get_processing_errors(limit=limit)
    return result
