from fastapi import APIRouter, HTTPException, status
from typing import Dict, List
from app.services.insight_aggregator import aggregator
from app.constants import DataSourceType
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasources", tags=["datasources"])


@router.get("/validate")
async def validate_datasources() -> Dict[str, bool]:
    try:
        validation_results = await aggregator.validate_all_connections()
        return validation_results
    except Exception as e:
        logger.error(f"Error validating datasources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/types")
async def get_datasource_types() -> List[str]:
    return [source_type.value for source_type in DataSourceType]


@router.get("/registered")
async def get_registered_datasources() -> Dict[str, str]:
    registered = {}
    for source_type, service in aggregator.services.items():
        registered[source_type] = service.__class__.__name__
    return registered
