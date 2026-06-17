from enum import Enum


class DataSourceType(str, Enum):
    DATABASE = "database"
    CLOUDWATCH = "cloudwatch"
    THIRD_PARTY_API = "third_party_api"
    S3 = "s3"
    DYNAMODB = "dynamodb"


class InsightStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


API_PREFIX = "/api/v1"
INSIGHTS_ROUTE = f"{API_PREFIX}/insights"
HEALTH_ROUTE = f"{API_PREFIX}/health"
DATASOURCES_ROUTE = f"{API_PREFIX}/datasources"

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2
