from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from loguru import logger
from app.core.storage import S3Storage
from app.models.schemas import Measurement
from sqlalchemy.orm import Session
from app.models.database import get_db
import uuid
from datetime import datetime

router = APIRouter()

@router.post("/upload", response_model=dict)
async def upload_images(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    height: float = None,
    db: Session = Depends(get_db),
):
    try:
        # Validate file types
        if not front_image.content_type.startswith("image/") or not side_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

        # Generate unique keys
        user_id = str(uuid.uuid4())
        front_key = f"uploads/{user_id}/front.jpg"
        side_key = f"uploads/{user_id}/side.jpg"

        # Upload to S3
        s3_storage = S3Storage()
        front_file = await front_image.read()
        side_file = await side_image.read()
        front_url = await s3_storage.upload_file(front_file, front_key)
        side_url = await s3_storage.upload_file(side_file, side_key)

        # Create measurement record using ORM
        new_measurement = Measurement(user_id=user_id)
        db.add(new_measurement)
        db.commit()
        db.refresh(new_measurement)

        # Lazy import to avoid circular imports
        from app.tasks.process_measurements import process_body_measurements
        process_body_measurements.delay(
            measurement_id=new_measurement.id,
            front_url=front_url,
            side_url=side_url,
            height=height,
        )

        logger.info(f"Measurement task triggered for user_id: {user_id}")
        return {"measurement_id": new_measurement.id, "status": "Processing"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to process upload")