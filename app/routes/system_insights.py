from fastapi import APIRouter, Query
from typing import Dict, Any
from app.services.system_insights_service import SystemInsightsService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/system", tags=["System Insights"])

system_service = SystemInsightsService()


@router.get("/dashboard")
async def get_system_health_dashboard() -> Dict[str, Any]:
    result = await system_service.get_system_health_dashboard()
    return result


@router.get("/broker-performance")
async def get_broker_performance(
    limit: int = Query(10, ge=1, le=50, description="Number of top brokers to return")
) -> Dict[str, Any]:
    result = await system_service.get_broker_performance(limit=limit)
    return result


@router.get("/contracts")
async def get_contract_summary() -> Dict[str, Any]:
    result = await system_service.get_contract_summary()
    return result


@router.get("/rule-execution")
async def get_rule_execution_stats(
    limit: int = Query(20, ge=1, le=100, description="Number of rules to return")
) -> Dict[str, Any]:
    result = await system_service.get_rule_execution_stats(limit=limit)
    return result


@router.get("/member-demographics")
async def get_member_demographics_summary() -> Dict[str, Any]:
    result = await system_service.get_member_demographics_summary()
    return result


@router.get("/recent-activity")
async def get_recent_activity(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back")
) -> Dict[str, Any]:
    result = await system_service.get_recent_activity(hours=hours)
    return result
