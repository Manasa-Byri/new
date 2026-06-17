from typing import Dict, Any, List
from datetime import datetime, timedelta
from pymongo import MongoClient
import logging
import os

logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")


class MongoDBProcessingInsightsService:
    
    def __init__(self):
        self.mongodb_url = MONGODB_URL
        self.client = None
        self.db = None
    
    def _get_connection(self):
        if not self.client:
            self.client = MongoClient(self.mongodb_url)
            self.db = self.client['edi_processor']
        return self.db
    
    async def get_processing_summary(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            
            member_records = db['member_processing_records']
            chunks = db['processing_chunks']
            audit_logs = db['processing_audit_log']
            
            summary = {
                "total_member_records": member_records.count_documents({}),
                "total_chunks": chunks.count_documents({}),
                "total_audit_logs": audit_logs.count_documents({})
            }
            
            if summary['total_member_records'] > 0:
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "total_processed": {"$sum": 1},
                            "completed": {"$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}},
                            "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}},
                            "pending": {"$sum": {"$cond": [{"$eq": ["$status", "pending"]}, 1, 0]}}
                        }
                    }
                ]
                
                result = list(member_records.aggregate(pipeline))
                if result:
                    summary.update(result[0])
                    summary.pop('_id', None)
                    
                    if summary['total_processed'] > 0:
                        summary['success_rate'] = round(
                            (summary['completed'] / summary['total_processed']) * 100, 2
                        )
            
            return {
                "success": True,
                "data": summary,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in get_processing_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_chunk_processing_stats(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['processing_chunks']
            
            pipeline = [
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "total_records": {"$sum": "$record_count"}
                    }
                },
                {
                    "$sort": {"count": -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            formatted_results = []
            for item in results:
                formatted_results.append({
                    "status": item['_id'],
                    "chunk_count": item['count'],
                    "total_records": item.get('total_records', 0)
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_chunk_processing_stats: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_member_processing_by_status(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['member_processing_records']
            
            pipeline = [
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$sort": {"count": -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            total_count = sum(item['count'] for item in results)
            
            formatted_results = []
            for item in results:
                percentage = round((item['count'] / total_count) * 100, 2) if total_count > 0 else 0
                formatted_results.append({
                    "status": item['_id'],
                    "count": item['count'],
                    "percentage": percentage
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_member_processing_by_status: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_processing_audit_timeline(self, hours: int = 24) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['processing_audit_log']
            
            time_threshold = datetime.now() - timedelta(hours=hours)
            
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": time_threshold}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                            "hour": {"$hour": "$timestamp"}
                        },
                        "event_count": {"$sum": 1}
                    }
                },
                {
                    "$sort": {"_id.date": -1, "_id.hour": -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            formatted_results = []
            for item in results:
                formatted_results.append({
                    "date": item['_id']['date'],
                    "hour": item['_id']['hour'],
                    "event_count": item['event_count']
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "period_hours": hours
            }
        except Exception as e:
            logger.error(f"Error in get_processing_audit_timeline: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_recent_member_records(self, limit: int = 20) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['member_processing_records']
            
            records = list(collection.find().sort("created_at", -1).limit(limit))
            
            formatted_results = []
            for record in records:
                record['_id'] = str(record['_id'])
                if 'created_at' in record and isinstance(record['created_at'], datetime):
                    record['created_at'] = record['created_at'].isoformat()
                if 'updated_at' in record and isinstance(record['updated_at'], datetime):
                    record['updated_at'] = record['updated_at'].isoformat()
                formatted_results.append(record)
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_recent_member_records: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_processing_errors(self, limit: int = 20) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['member_processing_records']
            
            query = {
                "status": "failed"
            }
            
            records = list(collection.find(query).sort("updated_at", -1).limit(limit))
            
            formatted_results = []
            for record in records:
                record['_id'] = str(record['_id'])
                if 'created_at' in record and isinstance(record['created_at'], datetime):
                    record['created_at'] = record['created_at'].isoformat()
                if 'updated_at' in record and isinstance(record['updated_at'], datetime):
                    record['updated_at'] = record['updated_at'].isoformat()
                formatted_results.append(record)
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_processing_errors: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def __del__(self):
        if self.client:
            self.client.close()
