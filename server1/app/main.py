from fastapi import FastAPI
from loguru import logger
from app.api.endpoints import uploads, measurements
from app.models.database import engine, Base
from app.core.logging import setup_logging

app = FastAPI(title="SmartSize AI Backend", version="1.0.0")

setup_logging()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting SmartSize AI Backend...")
    Base.metadata.create_all(bind=engine)

app.include_router(uploads.router, prefix="/api/v1")
app.include_router(measurements.router, prefix="/api/v1")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down SmartSize AI Backend...")