from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict
import time
from collections import defaultdict
from app.config import get_settings

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int = None, period: int = None):
        super().__init__(app)
        self.rate_limit = rate_limit or settings.API_RATE_LIMIT
        self.period = period or settings.API_RATE_LIMIT_PERIOD
        self.requests: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = request.client.host
        current_time = time.time()
        
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.period
        ]
        
        if len(self.requests[client_ip]) >= self.rate_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.rate_limit} requests per {self.period} seconds."
            )
        
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response
