from celery import Celery
from loguru import logger
import cv2
import numpy as np
from app.core.storage import S3Storage
from app.services.ai_processing import BodyMeasurementProcessor
from app.services.size_recommendation import SizeRecommender
from app.models.database import SessionLocal
from app.models.schemas import Measurement
import asyncio

app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
)

GROUND_TRUTH_MEASUREMENTS = {
    "chest": 102.5,
    "waist": 82.0,
    "hips": 92.0,
    "shoulder_width": 46.0,
    "arm_length": 62.0,
    "leg_length": 105.0,
    "inseam": 85.0,
    "neck": 38.0,
}
GROUND_TRUTH_SIZE = "L"

@app.task(bind=True, ignore_result=False)
def process_body_measurements(self, measurement_id: int, front_url: str, side_url: str, height: float = None):
    try:
        logger.info(f"Starting processing for measurement ID: {measurement_id}")
        s3_storage = S3Storage()
        front_data = asyncio.run(s3_storage.get_file(front_url.split('/')[-1]))
        if not front_data:
            raise ValueError(f"Failed to retrieve front image from {front_url}")
        front_image = cv2.imdecode(np.frombuffer(front_data, np.uint8), cv2.IMREAD_COLOR)
        if front_image is None:
            raise ValueError("Failed to decode front image")

        side_data = asyncio.run(s3_storage.get_file(side_url.split('/')[-1]))
        if not side_data:
            raise ValueError(f"Failed to retrieve side image from {side_url}")
        side_image = cv2.imdecode(np.frombuffer(side_data, np.uint8), cv2.IMREAD_COLOR)
        if side_image is None:
            raise ValueError("Failed to decode side image")

        processor = BodyMeasurementProcessor()
        front_landmarks = processor.detect_landmarks(front_image)
        side_landmarks = processor.detect_landmarks(side_image)
        depth_front = processor.estimate_depth(front_image)
        depth_side = processor.estimate_depth(side_image)
        measurements = processor.calculate_measurements(front_landmarks, side_landmarks, depth_front, depth_side, height)

        recommender = SizeRecommender()
        size = recommender.recommend_size(measurements)

        measurement_metrics = processor.validate_measurements(measurements, GROUND_TRUTH_MEASUREMENTS)
        size_metrics = recommender.validate_size(size, GROUND_TRUTH_SIZE)

        with SessionLocal() as db:
            update_data = {
                "chest": measurements.get("chest", 85.0),
                "waist": measurements.get("waist", 70.0),
                "hips": measurements.get("hips", 80.0),
                "shoulder_width": measurements.get("shoulder_width", 38.0),
                "arm_length": measurements.get("arm_length", 50.0),
                "leg_length": measurements.get("leg_length", 90.0),
                "inseam": measurements.get("inseam", 65.0),
                "neck": measurements.get("neck", 33.0),
            }
            db.query(Measurement).filter(Measurement.id == measurement_id).update(update_data)
            db.commit()
            logger.info(f"Measurement processed for ID: {measurement_id}, Size: {size}")

        return {
            "status": "success",
            "measurement_id": measurement_id,
            "size": size,
            "measurements": measurements,
            "metrics": measurement_metrics,
            "size_metrics": size_metrics,
        }
    except Exception as e:
        logger.error(f"Processing failed for measurement ID {measurement_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)