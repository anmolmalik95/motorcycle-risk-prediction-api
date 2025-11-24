from fastapi import FastAPI
from app.schemas.echo import EchoRequest
from app.api.v1.risk_endpoints import router as risk_router

app = FastAPI()

app.include_router(risk_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"Message": "Motorcycle Risk API is alive!"}

@app.get("/health")
def health_check():
    return {"Status": "ok"}
