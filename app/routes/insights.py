from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from app.models.schemas import (
    DataSourceQuery,
    MultiSourceRequest,
    InsightResponse,
    AggregatedInsightResponse,
    DatabaseQueryRequest,
    CloudWatchQueryRequest,
    ThirdPartyAPIRequest
)
from app.services.insight_aggregator import aggregator
from app.constants import DataSourceType
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


@router.post("/query", response_model=InsightResponse)
async def query_single_source(request: DataSourceQuery):
    try:
        result = await aggregator.fetch_from_source(request.type, request.query)
        
        return InsightResponse(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error in query_single_source: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/query/multi", response_model=AggregatedInsightResponse)
async def query_multiple_sources(request: MultiSourceRequest):
    try:
        sources = [
            {"type": source.type, "query": source.query}
            for source in request.sources
        ]
        
        result = await aggregator.fetch_from_multiple_sources(sources)
        
        return AggregatedInsightResponse(
            success=result.get("success", False),
            results=result.get("results", {}),
            total_sources=result.get("total_sources", 0)
        )
    except Exception as e:
        logger.error(f"Error in query_multiple_sources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/database", response_model=InsightResponse)
async def query_database(request: DatabaseQueryRequest):
    try:
        query = {
            "sql": request.sql,
            "params": request.params
        }
        
        result = await aggregator.fetch_from_source(DataSourceType.DATABASE, query)
        
        return InsightResponse(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error in query_database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/cloudwatch", response_model=InsightResponse)
async def query_cloudwatch(request: CloudWatchQueryRequest):
    try:
        query = request.model_dump()
        
        result = await aggregator.fetch_from_source(DataSourceType.CLOUDWATCH, query)
        
        return InsightResponse(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error in query_cloudwatch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/third-party", response_model=InsightResponse)
async def query_third_party(request: ThirdPartyAPIRequest):
    try:
        query = request.model_dump()
        
        result = await aggregator.fetch_from_source(DataSourceType.THIRD_PARTY_API, query)
        
        return InsightResponse(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error in query_third_party: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
