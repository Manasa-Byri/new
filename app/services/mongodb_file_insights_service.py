from typing import Dict, Any, List
from datetime import datetime, timedelta
from pymongo import MongoClient
import logging
import os

logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")


class MongoDBFileInsightsService:
    
    def __init__(self):
        self.mongodb_url = MONGODB_URL
        self.client = None
        self.db = None
    
    def _get_connection(self):
        if not self.client:
            self.client = MongoClient(self.mongodb_url)
            self.db = self.client['file_tracking']
        return self.db
    
    async def get_file_ingestion_summary(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            total_files = collection.count_documents({})
            
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_files": {"$sum": 1},
                        "total_size_bytes": {"$sum": "$file_size_bytes"},
                        "total_records": {"$sum": "$total_records"},
                        "total_members": {"$sum": "$members_total"},
                        "members_processed": {"$sum": "$members_processed"},
                        "members_succeeded": {"$sum": "$members_succeeded"},
                        "members_failed": {"$sum": "$members_failed"},
                        "total_chunks": {"$sum": "$total_chunks"},
                        "chunks_completed": {"$sum": "$chunks_completed"},
                        "chunks_failed": {"$sum": "$chunks_failed"}
                    }
                }
            ]
            
            result = list(collection.aggregate(pipeline))
            
            if result:
                summary = result[0]
                summary.pop('_id', None)
                
                if summary.get('total_members', 0) > 0:
                    summary['member_success_rate'] = round(
                        (summary['members_succeeded'] / summary['total_members']) * 100, 2
                    )
                else:
                    summary['member_success_rate'] = 0
                
                if summary.get('total_chunks', 0) > 0:
                    summary['chunk_success_rate'] = round(
                        (summary['chunks_completed'] / summary['total_chunks']) * 100, 2
                    )
                else:
                    summary['chunk_success_rate'] = 0
            else:
                summary = {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_records": 0,
                    "total_members": 0,
                    "members_processed": 0,
                    "members_succeeded": 0,
                    "members_failed": 0,
                    "member_success_rate": 0,
                    "total_chunks": 0,
                    "chunks_completed": 0,
                    "chunks_failed": 0,
                    "chunk_success_rate": 0
                }
            
            return {
                "success": True,
                "data": summary,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in get_file_ingestion_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_files_by_status(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            pipeline = [
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "total_records": {"$sum": "$total_records"},
                        "total_members": {"$sum": "$members_total"}
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
                    "file_count": item['count'],
                    "total_records": item['total_records'],
                    "total_members": item['total_members']
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_files_by_status: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_files_by_source_system(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            pipeline = [
                {
                    "$group": {
                        "_id": "$source_system",
                        "file_count": {"$sum": 1},
                        "total_size_bytes": {"$sum": "$file_size_bytes"},
                        "total_records": {"$sum": "$total_records"},
                        "avg_file_size": {"$avg": "$file_size_bytes"}
                    }
                },
                {
                    "$sort": {"file_count": -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            formatted_results = []
            for item in results:
                formatted_results.append({
                    "source_system": item['_id'],
                    "file_count": item['file_count'],
                    "total_size_bytes": item['total_size_bytes'],
                    "total_records": item['total_records'],
                    "avg_file_size_bytes": round(item['avg_file_size'], 2) if item['avg_file_size'] else 0
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_files_by_source_system: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_file_format_distribution(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            pipeline = [
                {
                    "$group": {
                        "_id": "$file_format",
                        "count": {"$sum": 1},
                        "total_records": {"$sum": "$total_records"}
                    }
                },
                {
                    "$sort": {"count": -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            total_files = sum(item['count'] for item in results)
            
            formatted_results = []
            for item in results:
                percentage = round((item['count'] / total_files) * 100, 2) if total_files > 0 else 0
                formatted_results.append({
                    "file_format": item['_id'],
                    "count": item['count'],
                    "percentage": percentage,
                    "total_records": item['total_records']
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_file_format_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_processing_performance(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            pipeline = [
                {
                    "$match": {
                        "created_at": {"$exists": True},
                        "updated_at": {"$exists": True}
                    }
                },
                {
                    "$project": {
                        "file_name": 1,
                        "status": 1,
                        "total_records": 1,
                        "members_total": 1,
                        "members_processed": 1,
                        "members_succeeded": 1,
                        "members_failed": 1,
                        "chunks_completed": 1,
                        "chunks_failed": 1,
                        "total_chunks": 1,
                        "error_count": 1,
                        "retry_count": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "processing_time_seconds": {
                            "$divide": [
                                {"$subtract": ["$updated_at", "$created_at"]},
                                1000
                            ]
                        }
                    }
                },
                {
                    "$sort": {"updated_at": -1}
                },
                {
                    "$limit": 20
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            formatted_results = []
            for item in results:
                item['_id'] = str(item['_id'])
                item['created_at'] = item['created_at'].isoformat() if isinstance(item.get('created_at'), datetime) else item.get('created_at')
                item['updated_at'] = item['updated_at'].isoformat() if isinstance(item.get('updated_at'), datetime) else item.get('updated_at')
                
                if item.get('members_total', 0) > 0:
                    item['success_rate'] = round((item.get('members_succeeded', 0) / item['members_total']) * 100, 2)
                else:
                    item['success_rate'] = 0
                
                formatted_results.append(item)
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_processing_performance: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_recent_uploads(self, hours: int = 24) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            time_threshold = datetime.now() - timedelta(hours=hours)
            
            query = {
                "uploaded_at": {"$gte": time_threshold}
            }
            
            files = list(collection.find(query).sort("uploaded_at", -1).limit(50))
            
            formatted_results = []
            for file in files:
                file['_id'] = str(file['_id'])
                file['uploaded_at'] = file['uploaded_at'].isoformat() if isinstance(file.get('uploaded_at'), datetime) else file.get('uploaded_at')
                file['created_at'] = file['created_at'].isoformat() if isinstance(file.get('created_at'), datetime) else file.get('created_at')
                file['updated_at'] = file['updated_at'].isoformat() if isinstance(file.get('updated_at'), datetime) else file.get('updated_at')
                formatted_results.append(file)
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results),
                "period_hours": hours
            }
        except Exception as e:
            logger.error(f"Error in get_recent_uploads: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_error_analysis(self) -> Dict[str, Any]:
        try:
            db = self._get_connection()
            collection = db['ingestion_files']
            
            pipeline = [
                {
                    "$match": {
                        "error_count": {"$gt": 0}
                    }
                },
                {
                    "$group": {
                        "_id": "$status",
                        "files_with_errors": {"$sum": 1},
                        "total_errors": {"$sum": "$error_count"},
                        "total_retries": {"$sum": "$retry_count"},
                        "avg_errors_per_file": {"$avg": "$error_count"}
                    }
                },
                {
                    "$sort": {"total_errors": -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            formatted_results = []
            for item in results:
                formatted_results.append({
                    "status": item['_id'],
                    "files_with_errors": item['files_with_errors'],
                    "total_errors": item['total_errors'],
                    "total_retries": item['total_retries'],
                    "avg_errors_per_file": round(item['avg_errors_per_file'], 2)
                })
            
            return {
                "success": True,
                "data": formatted_results,
                "count": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in get_error_analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def __del__(self):
        if self.client:
            self.client.close()
