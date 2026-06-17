from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.services.base_service import BaseDataSourceService
from app.database import get_db
import logging

logger = logging.getLogger(__name__)


class DatabaseService(BaseDataSourceService):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_session: Optional[Session] = None
    
    async def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sql_query = query.get("sql")
            params = query.get("params", {})
            
            if not sql_query:
                raise ValueError("SQL query is required")
            
            db = next(get_db())
            result = db.execute(text(sql_query), params)
            
            rows = []
            for row in result:
                rows.append(dict(row._mapping))
            
            return {
                "success": True,
                "data": rows,
                "count": len(rows)
            }
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def validate_connection(self) -> bool:
        try:
            db = next(get_db())
            db.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection validation failed: {str(e)}")
            return False
    
    async def execute_query(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        query = {"sql": sql, "params": params or {}}
        result = await self.fetch_data(query)
        return result.get("data", [])
