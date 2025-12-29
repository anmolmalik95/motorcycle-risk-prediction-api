from fastapi import FastAPI
from app.schemas.echo import EchoRequest
from app.api.v1.risk_endpoints import router as risk_router
import logging
import os

APP_VERSION = os.getenv("APP_VERSION", "dev")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("motorcycle-risk-api")

app = FastAPI()

app.include_router(risk_router, prefix="/api/v1")

@app.on_event("startup")
def on_startup():
    logger.info(f"Motorcycle Risk API starting up | version={APP_VERSION}")

@app.get("/")
def read_root():
    return {"Message": "Motorcycle Risk API is alive!"}

@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "ok"}
