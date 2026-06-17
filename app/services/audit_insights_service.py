from typing import Dict, Any, List
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "10.125.81.118",
    "database": "testdb",
    "user": "postgres",
    "password": "DEP@123",
    "port": 5432
}


class AuditInsightsService:
    
    def __init__(self):
        self.db_config = DB_CONFIG
    
    def _get_connection(self):
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
    
    async def get_api_usage_summary(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT ip_address) as unique_ips,
                    COUNT(CASE WHEN response_status >= 200 AND response_status < 300 THEN 1 END) as successful_requests,
                    COUNT(CASE WHEN response_status >= 400 AND response_status < 500 THEN 1 END) as client_errors,
                    COUNT(CASE WHEN response_status >= 500 THEN 1 END) as server_errors,
                    MIN(created_at) as first_request,
                    MAX(created_at) as last_request
                FROM audit_logs
            """)
            
            summary = dict(cursor.fetchone())
            
            summary['first_request'] = summary['first_request'].isoformat() if summary['first_request'] else None
            summary['last_request'] = summary['last_request'].isoformat() if summary['last_request'] else None
            
            if summary['total_requests'] > 0:
                summary['success_rate'] = round(
                    (summary['successful_requests'] / summary['total_requests']) * 100, 2
                )
            else:
                summary['success_rate'] = 0
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": summary
            }
        except Exception as e:
            logger.error(f"Error in get_api_usage_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_top_endpoints(self, limit: int = 10) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    request_path,
                    request_method,
                    COUNT(*) as request_count,
                    COUNT(CASE WHEN response_status >= 200 AND response_status < 300 THEN 1 END) as successful,
                    COUNT(CASE WHEN response_status >= 400 THEN 1 END) as errors,
                    AVG(CASE WHEN response_status >= 200 AND response_status < 300 THEN 1 ELSE 0 END) * 100 as success_rate
                FROM audit_logs
                WHERE request_path IS NOT NULL
                GROUP BY request_path, request_method
                ORDER BY request_count DESC
                LIMIT %s
            """, (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                row['success_rate'] = round(row['success_rate'], 2) if row['success_rate'] else 0
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_top_endpoints: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_request_volume_by_hour(self, days: int = 7) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    EXTRACT(HOUR FROM created_at) as hour,
                    COUNT(*) as request_count
                FROM audit_logs
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(created_at), EXTRACT(HOUR FROM created_at)
                ORDER BY date DESC, hour
            """, (days,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                row['date'] = row['date'].isoformat() if row['date'] else None
                row['hour'] = int(row['hour']) if row['hour'] else 0
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "period_days": days
            }
        except Exception as e:
            logger.error(f"Error in get_request_volume_by_hour: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_response_status_distribution(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    response_status,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
                FROM audit_logs
                WHERE response_status IS NOT NULL
                GROUP BY response_status
                ORDER BY count DESC
            """)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_response_status_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_top_ip_addresses(self, limit: int = 10) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    ip_address,
                    COUNT(*) as request_count,
                    COUNT(DISTINCT DATE(created_at)) as active_days,
                    MAX(created_at) as last_activity
                FROM audit_logs
                WHERE ip_address IS NOT NULL
                GROUP BY ip_address
                ORDER BY request_count DESC
                LIMIT %s
            """, (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                row['last_activity'] = row['last_activity'].isoformat() if row['last_activity'] else None
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_top_ip_addresses: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_member_audit_summary(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_changes,
                    COUNT(DISTINCT member_id) as members_affected,
                    MIN(changed_at) as first_change,
                    MAX(changed_at) as last_change
                FROM member_audit
            """)
            
            summary = dict(cursor.fetchone())
            
            summary['first_change'] = summary['first_change'].isoformat() if summary['first_change'] else None
            summary['last_change'] = summary['last_change'].isoformat() if summary['last_change'] else None
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": summary
            }
        except Exception as e:
            logger.error(f"Error in get_member_audit_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
