from typing import Any, Dict, List, Optional
from app.services.database_service import DatabaseService
from app.services.cloudwatch_service import CloudWatchService
from app.services.third_party_service import ThirdPartyAPIService
from app.constants import DataSourceType
import logging
import asyncio

logger = logging.getLogger(__name__)


class InsightAggregator:
    
    def __init__(self):
        self.services = {}
    
    def register_service(self, source_type: DataSourceType, service):
        self.services[source_type] = service
        logger.info(f"Registered service for {source_type}")
    
    async def fetch_from_source(
        self, 
        source_type: DataSourceType, 
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        service = self.services.get(source_type)
        
        if not service:
            return {
                "success": False,
                "error": f"No service registered for {source_type}",
                "data": None
            }
        
        try:
            result = await service.fetch_data(query)
            return result
        except Exception as e:
            logger.error(f"Error fetching from {source_type}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def fetch_from_multiple_sources(
        self, 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tasks = []
        source_names = []
        
        for source in sources:
            source_type = source.get("type")
            query = source.get("query", {})
            
            if source_type:
                tasks.append(self.fetch_from_source(DataSourceType(source_type), query))
                source_names.append(source_type)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        aggregated_results = {}
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                aggregated_results[name] = {
                    "success": False,
                    "error": str(result),
                    "data": None
                }
            else:
                aggregated_results[name] = result
        
        return {
            "success": True,
            "results": aggregated_results,
            "total_sources": len(sources)
        }
    
    async def validate_all_connections(self) -> Dict[str, bool]:
        validation_results = {}
        
        for source_type, service in self.services.items():
            try:
                is_valid = await service.validate_connection()
                validation_results[source_type] = is_valid
            except Exception as e:
                logger.error(f"Validation failed for {source_type}: {str(e)}")
                validation_results[source_type] = False
        
        return validation_results


aggregator = InsightAggregator()
