from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseDataSourceService(ABC):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
    
    @abstractmethod
    async def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        pass
    
    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "service_name": self.__class__.__name__,
            "config_keys": list(self.config.keys())
        }
