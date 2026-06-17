from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
from app.services.base_service import BaseDataSourceService
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class CloudWatchService(BaseDataSourceService):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            session_config = {
                "region_name": self.config.get("region", settings.AWS_REGION)
            }
            
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                session_config["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
                session_config["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            
            self.client = boto3.client("logs", **session_config)
        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch client: {str(e)}")
    
    async def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            log_group = query.get("log_group", settings.CLOUDWATCH_LOG_GROUP)
            start_time = query.get("start_time")
            end_time = query.get("end_time")
            filter_pattern = query.get("filter_pattern", "")
            limit = query.get("limit", 100)
            
            if not log_group:
                raise ValueError("Log group is required")
            
            if not start_time:
                start_time = datetime.now() - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now()
            
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            params = {
                "logGroupName": log_group,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit
            }
            
            if filter_pattern:
                params["filterPattern"] = filter_pattern
            
            response = self.client.filter_log_events(**params)
            
            events = response.get("events", [])
            
            return {
                "success": True,
                "data": events,
                "count": len(events),
                "next_token": response.get("nextToken")
            }
        except ClientError as e:
            logger.error(f"CloudWatch query failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
        except Exception as e:
            logger.error(f"Unexpected error in CloudWatch service: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def validate_connection(self) -> bool:
        try:
            self.client.describe_log_groups(limit=1)
            return True
        except Exception as e:
            logger.error(f"CloudWatch connection validation failed: {str(e)}")
            return False
    
    async def get_log_groups(self) -> List[str]:
        try:
            response = self.client.describe_log_groups()
            return [lg["logGroupName"] for lg in response.get("logGroups", [])]
        except Exception as e:
            logger.error(f"Failed to get log groups: {str(e)}")
            return []
