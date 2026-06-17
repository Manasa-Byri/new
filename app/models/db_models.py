from sqlalchemy import Column, Integer, String, DateTime, JSON, Enum as SQLEnum
from datetime import datetime
from app.database import Base
from app.constants import DataSourceType, InsightStatus
import enum


class InsightLog(Base):
    __tablename__ = "insight_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(SQLEnum(DataSourceType), nullable=False)
    query = Column(JSON, nullable=False)
    status = Column(SQLEnum(InsightStatus), default=InsightStatus.PENDING)
    result = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<InsightLog(id={self.id}, source_type={self.source_type}, status={self.status})>"


class DataSourceConfig(Base):
    __tablename__ = "datasource_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    source_type = Column(SQLEnum(DataSourceType), nullable=False)
    config = Column(JSON, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<DataSourceConfig(id={self.id}, name={self.name}, type={self.source_type})>"
