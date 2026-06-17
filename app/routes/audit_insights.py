from fastapi import APIRouter, Query
from typing import Dict, Any
from app.services.audit_insights_service import AuditInsightsService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/audit", tags=["Audit Insights"])

audit_service = AuditInsightsService()


@router.get("/api-usage")
async def get_api_usage_summary() -> Dict[str, Any]:
    result = await audit_service.get_api_usage_summary()
    return result


@router.get("/top-endpoints")
async def get_top_endpoints(
    limit: int = Query(10, ge=1, le=50, description="Number of top endpoints to return")
) -> Dict[str, Any]:
    result = await audit_service.get_top_endpoints(limit=limit)
    return result


@router.get("/request-volume")
async def get_request_volume_by_hour(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
) -> Dict[str, Any]:
    result = await audit_service.get_request_volume_by_hour(days=days)
    return result


@router.get("/response-status")
async def get_response_status_distribution() -> Dict[str, Any]:
    result = await audit_service.get_response_status_distribution()
    return result


@router.get("/top-ips")
async def get_top_ip_addresses(
    limit: int = Query(10, ge=1, le=50, description="Number of top IPs to return")
) -> Dict[str, Any]:
    result = await audit_service.get_top_ip_addresses(limit=limit)
    return result


@router.get("/member-audit")
async def get_member_audit_summary() -> Dict[str, Any]:
    result = await audit_service.get_member_audit_summary()
    return result
