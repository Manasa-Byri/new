from fastapi import APIRouter, Query
from typing import Dict, Any
from app.services.csv_insurance_insights_service import CSVInsuranceInsightsService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/csv", tags=["CSV Insurance Insights"])

csv_service = CSVInsuranceInsightsService()


@router.get("/consolidated-summary")
async def get_consolidated_summary() -> Dict[str, Any]:
    result = await csv_service.get_consolidated_summary()
    return result


@router.get("/membership/summary")
async def get_membership_summary() -> Dict[str, Any]:
    result = await csv_service.get_membership_summary()
    return result


@router.get("/membership/by-state")
async def get_membership_by_state() -> Dict[str, Any]:
    result = await csv_service.get_membership_by_state()
    return result


@router.get("/membership/age-demographics")
async def get_age_demographics() -> Dict[str, Any]:
    result = await csv_service.get_age_demographics()
    return result


@router.get("/membership/member-types")
async def get_member_type_distribution() -> Dict[str, Any]:
    result = await csv_service.get_member_type_distribution()
    return result


@router.get("/membership/family-size")
async def get_family_size_distribution() -> Dict[str, Any]:
    result = await csv_service.get_family_size_distribution()
    return result


@router.get("/plans/distribution")
async def get_plan_distribution() -> Dict[str, Any]:
    result = await csv_service.get_plan_distribution()
    return result


@router.get("/plans/by-state")
async def get_plan_by_state() -> Dict[str, Any]:
    result = await csv_service.get_plan_by_state()
    return result


@router.get("/plans/hmo-vs-ppo")
async def get_hmo_vs_ppo_analysis() -> Dict[str, Any]:
    result = await csv_service.get_hmo_vs_ppo_analysis()
    return result


@router.get("/cancellations/reasons")
async def get_cancellation_reasons(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of reasons to return")
) -> Dict[str, Any]:
    result = await csv_service.get_cancellation_reasons(limit=limit)
    return result


@router.get("/cancellations/by-plan-type")
async def get_cancellation_by_plan_type() -> Dict[str, Any]:
    result = await csv_service.get_cancellation_by_plan_type()
    return result


@router.get("/business/segment-analysis")
async def get_business_segment_analysis() -> Dict[str, Any]:
    result = await csv_service.get_business_segment_analysis()
    return result


@router.get("/business/exchange-distribution")
async def get_exchange_distribution() -> Dict[str, Any]:
    result = await csv_service.get_exchange_distribution()
    return result


@router.get("/providers/performance")
async def get_provider_performance(
    limit: int = Query(20, ge=1, le=100, description="Number of top providers to return")
) -> Dict[str, Any]:
    result = await csv_service.get_provider_performance(limit=limit)
    return result


@router.post("/admin/reload-data")
async def reload_data() -> Dict[str, Any]:
    result = await csv_service.reload_data()
    return result
