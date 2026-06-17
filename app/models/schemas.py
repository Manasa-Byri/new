from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
from app.constants import DataSourceType, InsightStatus


class DataSourceQuery(BaseModel):
    type: DataSourceType
    query: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "database",
                "query": {
                    "sql": "SELECT * FROM users LIMIT 10",
                    "params": {}
                }
            }
        }


class MultiSourceRequest(BaseModel):
    sources: List[DataSourceQuery]
    
    class Config:
        json_schema_extra = {
            "example": {
                "sources": [
                    {
                        "type": "database",
                        "query": {"sql": "SELECT COUNT(*) FROM users"}
                    },
                    {
                        "type": "cloudwatch",
                        "query": {"log_group": "/aws/lambda/my-function", "limit": 50}
                    }
                ]
            }
        }


class InsightResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"count": 100},
                "error": None,
                "timestamp": "2026-05-26T07:50:00"
            }
        }


class AggregatedInsightResponse(BaseModel):
    success: bool
    results: Dict[str, Any]
    total_sources: int
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Optional[Dict[str, bool]] = None


class DatabaseQueryRequest(BaseModel):
    sql: str
    params: Optional[Dict[str, Any]] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "sql": "SELECT * FROM users WHERE id = :user_id",
                "params": {"user_id": 123}
            }
        }


class CloudWatchQueryRequest(BaseModel):
    log_group: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    filter_pattern: Optional[str] = ""
    limit: int = 100
    
    class Config:
        json_schema_extra = {
            "example": {
                "log_group": "/aws/lambda/my-function",
                "start_time": "2026-05-26T06:50:00",
                "end_time": "2026-05-26T07:50:00",
                "filter_pattern": "ERROR",
                "limit": 100
            }
        }


class ThirdPartyAPIRequest(BaseModel):
    endpoint: str
    method: str = "GET"
    params: Optional[Dict[str, Any]] = {}
    body: Optional[Dict[str, Any]] = {}
    headers: Optional[Dict[str, str]] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "endpoint": "users/123",
                "method": "GET",
                "params": {},
                "headers": {"Authorization": "Bearer token"}
            }
        }


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
