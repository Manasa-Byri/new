from typing import Any, Dict, Optional
import httpx
from app.services.base_service import BaseDataSourceService
from app.constants import REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
import logging
import asyncio

logger = logging.getLogger(__name__)


class ThirdPartyAPIService(BaseDataSourceService):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = self.config.get("base_url", "")
        self.headers = self.config.get("headers", {})
        self.timeout = self.config.get("timeout", REQUEST_TIMEOUT)
    
    async def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = query.get("endpoint", "")
        method = query.get("method", "GET").upper()
        params = query.get("params", {})
        body = query.get("body", {})
        custom_headers = query.get("headers", {})
        
        url = f"{self.base_url}/{endpoint}".rstrip("/")
        headers = {**self.headers, **custom_headers}
        
        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        response = await client.get(url, params=params, headers=headers)
                    elif method == "POST":
                        response = await client.post(url, json=body, params=params, headers=headers)
                    elif method == "PUT":
                        response = await client.put(url, json=body, params=params, headers=headers)
                    elif method == "DELETE":
                        response = await client.delete(url, params=params, headers=headers)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    response.raise_for_status()
                    
                    return {
                        "success": True,
                        "data": response.json(),
                        "status_code": response.status_code
                    }
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": f"HTTP {e.response.status_code}: {str(e)}",
                        "data": None
                    }
                await asyncio.sleep(RETRY_BACKOFF ** attempt)
            except Exception as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "data": None
                    }
                await asyncio.sleep(RETRY_BACKOFF ** attempt)
    
    async def validate_connection(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(self.base_url, headers=self.headers)
                return response.status_code < 500
        except Exception as e:
            logger.error(f"Third-party API connection validation failed: {str(e)}")
            return False
