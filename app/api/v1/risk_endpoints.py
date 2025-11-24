from fastapi import APIRouter
from app.schemas.echo import EchoRequest
from app.schemas.risk import RiskRequest, RiskResponse
from app.services.risk_service import predict_risk as predict_risk_service

router = APIRouter(
    prefix="/risk",
    tags=["risk"],
)

@router.get("/hello/{name}")
def say_hello(name):
    return {"messsage": f"Hello {name}!"}

@router.post("/echo")
def echo(payload:EchoRequest):
    return {"received":payload.message, "also": payload.secret}

@router.post("/predict-risk", response_model=RiskResponse)
def predict(payload: RiskRequest) -> RiskResponse:
    result = predict_risk_service(payload)
    return result
