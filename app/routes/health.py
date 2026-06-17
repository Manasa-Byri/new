from fastapi import APIRouter
from app.models.schemas import HealthCheckResponse
from app.config import get_settings
from app.services.insight_aggregator import aggregator
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION
    )


@router.get("/detailed", response_model=HealthCheckResponse)
async def detailed_health_check():
    try:
        service_status = await aggregator.validate_all_connections()
        
        return HealthCheckResponse(
            status="healthy",
            version=settings.APP_VERSION,
            services=service_status
        )
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return HealthCheckResponse(
            status="degraded",
            version=settings.APP_VERSION,
            services={}
        )
