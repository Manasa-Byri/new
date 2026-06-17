from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from app.config import get_settings
from app.database import init_db
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.routes import insights, health, datasources, enrollment_insights, audit_insights, system_insights, mongodb_file_insights, mongodb_processing_insights, csv_insurance_insights, ml_anomaly_detection
from app.routes import predict_anomaly
from app.api import router as score_router
from app.services.insight_aggregator import aggregator
from app.services.database_service import DatabaseService
from app.services.cloudwatch_service import CloudWatchService
from app.services.third_party_service import ThirdPartyAPIService
from app.constants import DataSourceType
import logging

settings = get_settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {str(e)}")
    
    try:
        db_service = DatabaseService()
        aggregator.register_service(DataSourceType.DATABASE, db_service)
        logger.info("Database service registered")
    except Exception as e:
        logger.warning(f"Database service registration failed: {str(e)}")
    
    try:
        cloudwatch_service = CloudWatchService()
        aggregator.register_service(DataSourceType.CLOUDWATCH, cloudwatch_service)
        logger.info("CloudWatch service registered")
    except Exception as e:
        logger.warning(f"CloudWatch service registration failed: {str(e)}")
    
    try:
        third_party_service = ThirdPartyAPIService()
        aggregator.register_service(DataSourceType.THIRD_PARTY_API, third_party_service)
        logger.info("Third-party API service registered")
    except Exception as e:
        logger.warning(f"Third-party API service registration failed: {str(e)}")
    
    logger.info("Application startup complete")
    
    yield
    
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(health.router, prefix="/api/v1")
app.include_router(insights.router, prefix="/api/v1")
app.include_router(datasources.router, prefix="/api/v1")
app.include_router(enrollment_insights.router, prefix="/api/v1")
app.include_router(audit_insights.router, prefix="/api/v1")
app.include_router(system_insights.router, prefix="/api/v1")
app.include_router(mongodb_file_insights.router, prefix="/api/v1")
app.include_router(mongodb_processing_insights.router, prefix="/api/v1")
app.include_router(csv_insurance_insights.router, prefix="/api/v1")
app.include_router(ml_anomaly_detection.router, prefix="/api/v1")
app.include_router(predict_anomaly.router, prefix="/api/v1")
app.include_router(score_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }
