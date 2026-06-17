from typing import Dict, Any
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


class SystemInsightsService:
    
    def __init__(self):
        self.db_config = DB_CONFIG
    
    def _get_connection(self):
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
    
    async def get_system_health_dashboard(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM members) as total_members,
                    (SELECT COUNT(*) FROM canonical_enrollments) as total_enrollments,
                    (SELECT COUNT(*) FROM contracts WHERE status = 'active') as active_contracts,
                    (SELECT COUNT(*) FROM brokers WHERE status = 'active') as active_brokers,
                    (SELECT COUNT(*) FROM enrollment_files_pg) as files_processed,
                    (SELECT COUNT(*) FROM audit_logs WHERE created_at >= NOW() - INTERVAL '24 hours') as requests_24h,
                    (SELECT COUNT(*) FROM canonical_enrollments WHERE created_at >= NOW() - INTERVAL '24 hours') as enrollments_24h,
                    (SELECT COUNT(*) FROM canonical_enrollments WHERE validation_status = 'failed' AND created_at >= NOW() - INTERVAL '24 hours') as failed_validations_24h
            """)
            
            dashboard = dict(cursor.fetchone())
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": dashboard,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in get_system_health_dashboard: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_broker_performance(self, limit: int = 10) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    b.broker_id,
                    b.name as broker_name,
                    b.agency_name,
                    b.status,
                    COUNT(ce.id) as total_enrollments,
                    COUNT(CASE WHEN ce.validation_status = 'passed' THEN 1 END) as passed_enrollments,
                    COUNT(CASE WHEN ce.validation_status = 'failed' THEN 1 END) as failed_enrollments
                FROM brokers b
                LEFT JOIN canonical_enrollments ce ON ce.canonical_data->>'broker'->>'broker_id' = b.broker_id
                GROUP BY b.broker_id, b.name, b.agency_name, b.status
                ORDER BY total_enrollments DESC
                LIMIT %s
            """, (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                if row['total_enrollments'] > 0:
                    row['success_rate'] = round((row['passed_enrollments'] / row['total_enrollments']) * 100, 2)
                else:
                    row['success_rate'] = 0
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_broker_performance: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_contract_summary(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    state_code,
                    status,
                    COUNT(*) as contract_count,
                    COUNT(DISTINCT sponsor_name) as unique_sponsors,
                    COUNT(DISTINCT payer_name) as unique_payers
                FROM contracts
                GROUP BY state_code, status
                ORDER BY state_code, status
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
            logger.error(f"Error in get_contract_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_rule_execution_stats(self, limit: int = 20) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    rule_unique_id,
                    error_code,
                    error_message,
                    action,
                    severity,
                    COUNT(*) as execution_count
                FROM business_rules
                GROUP BY rule_unique_id, error_code, error_message, action, severity
                ORDER BY execution_count DESC
                LIMIT %s
            """, (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_rule_execution_stats: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_member_demographics_summary(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_members,
                    COUNT(DISTINCT state_code) as states_represented,
                    COUNT(CASE WHEN validation_status = 'passed' THEN 1 END) as validated_members,
                    COUNT(CASE WHEN hold_flag = true THEN 1 END) as members_on_hold
                FROM canonical_enrollments
            """)
            
            summary = dict(cursor.fetchone())
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": summary
            }
        except Exception as e:
            logger.error(f"Error in get_member_demographics_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    'enrollment' as activity_type,
                    COUNT(*) as count,
                    MAX(created_at) as last_activity
                FROM canonical_enrollments
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                UNION ALL
                SELECT 
                    'api_request' as activity_type,
                    COUNT(*) as count,
                    MAX(created_at) as last_activity
                FROM audit_logs
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                UNION ALL
                SELECT 
                    'member_change' as activity_type,
                    COUNT(*) as count,
                    MAX(changed_at) as last_activity
                FROM member_audit
                WHERE changed_at >= NOW() - INTERVAL '%s hours'
            """, (hours, hours, hours))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                row['last_activity'] = row['last_activity'].isoformat() if row['last_activity'] else None
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "period_hours": hours
            }
        except Exception as e:
            logger.error(f"Error in get_recent_activity: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
