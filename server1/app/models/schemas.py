from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from app.models.database import Base

class Measurement(Base):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    chest = Column(Float, nullable=True)
    waist = Column(Float, nullable=True)
    hips = Column(Float, nullable=True)
    shoulder_width = Column(Float, nullable=True)
    arm_length = Column(Float, nullable=True)
    leg_length = Column(Float, nullable=True)
    inseam = Column(Float, nullable=True)
    neck = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class MeasurementCreate(BaseModel):
    user_id: str
    height: Optional[float] = None

class MeasurementResponse(BaseModel):
    id: int
    user_id: str
    chest: Optional[float]
    waist: Optional[float]
    hips: Optional[float]
    shoulder_width: Optional[float]
    arm_length: Optional[float]
    leg_length: Optional[float]
    inseam: Optional[float]
    neck: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True