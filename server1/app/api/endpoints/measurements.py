from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.schemas import MeasurementResponse, Measurement
from loguru import logger
from datetime import datetime

router = APIRouter()

@router.get("/measurements/{measurement_id}", response_model=MeasurementResponse)
async def get_measurement(measurement_id: int, db: Session = Depends(get_db)):
    try:
        measurement = db.query(Measurement).filter(Measurement.id == measurement_id).first()
        if not measurement:
            raise HTTPException(status_code=404, detail="Measurement not found")
        if measurement.created_at is None:
            measurement.created_at = datetime.utcnow()
            db.commit()
        return measurement
    except Exception as e:
        logger.error(f"Failed to retrieve measurement: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")