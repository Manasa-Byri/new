from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
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


class EnrollmentInsightsService:
    
    def __init__(self):
        self.db_config = DB_CONFIG
    
    def _get_connection(self):
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
    
    async def get_enrollment_summary(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_enrollments,
                    COUNT(DISTINCT member_id) as unique_members,
                    COUNT(CASE WHEN validation_status = 'passed' THEN 1 END) as passed_validations,
                    COUNT(CASE WHEN validation_status = 'failed' THEN 1 END) as failed_validations,
                    COUNT(CASE WHEN hold_flag = true THEN 1 END) as on_hold,
                    COUNT(DISTINCT state_code) as states_covered,
                    COUNT(DISTINCT file_name) as files_processed
                FROM canonical_enrollments
            """)
            
            summary = dict(cursor.fetchone())
            
            if summary['total_enrollments'] > 0:
                summary['pass_rate'] = round(
                    (summary['passed_validations'] / summary['total_enrollments']) * 100, 2
                )
            else:
                summary['pass_rate'] = 0
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": summary,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in get_enrollment_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_enrollment_by_state(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    state_code,
                    COUNT(*) as enrollment_count,
                    COUNT(CASE WHEN validation_status = 'passed' THEN 1 END) as passed,
                    COUNT(CASE WHEN validation_status = 'failed' THEN 1 END) as failed
                FROM canonical_enrollments
                WHERE state_code IS NOT NULL
                GROUP BY state_code
                ORDER BY enrollment_count DESC
            """)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                if row['enrollment_count'] > 0:
                    row['pass_rate'] = round((row['passed'] / row['enrollment_count']) * 100, 2)
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_enrollment_by_state: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_enrollment_trends(self, days: int = 30) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as enrollment_count,
                    COUNT(CASE WHEN validation_status = 'passed' THEN 1 END) as passed,
                    COUNT(CASE WHEN validation_status = 'failed' THEN 1 END) as failed
                FROM canonical_enrollments
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, (days,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                row['date'] = row['date'].isoformat() if row['date'] else None
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "period_days": days
            }
        except Exception as e:
            logger.error(f"Error in get_enrollment_trends: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_validation_errors(self, limit: int = 20) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    error_message,
                    COUNT(*) as occurrence_count,
                    COUNT(DISTINCT member_id) as affected_members
                FROM canonical_enrollments
                WHERE error_message IS NOT NULL 
                  AND error_message != 'None'
                  AND validation_status = 'failed'
                GROUP BY error_message
                ORDER BY occurrence_count DESC
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
            logger.error(f"Error in get_validation_errors: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_maintenance_type_distribution(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    maintenance_type_code,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
                FROM canonical_enrollments
                WHERE maintenance_type_code IS NOT NULL
                GROUP BY maintenance_type_code
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
            logger.error(f"Error in get_maintenance_type_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_file_processing_stats(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    file_name,
                    COUNT(*) as records_processed,
                    COUNT(CASE WHEN validation_status = 'passed' THEN 1 END) as passed,
                    COUNT(CASE WHEN validation_status = 'failed' THEN 1 END) as failed,
                    MIN(created_at) as first_record_time,
                    MAX(created_at) as last_record_time
                FROM canonical_enrollments
                WHERE file_name IS NOT NULL
                GROUP BY file_name
                ORDER BY last_record_time DESC
                LIMIT 20
            """)
            
            results = [dict(row) for row in cursor.fetchall()]
            
            for row in results:
                row['first_record_time'] = row['first_record_time'].isoformat() if row['first_record_time'] else None
                row['last_record_time'] = row['last_record_time'].isoformat() if row['last_record_time'] else None
                if row['records_processed'] > 0:
                    row['pass_rate'] = round((row['passed'] / row['records_processed']) * 100, 2)
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_file_processing_stats: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
