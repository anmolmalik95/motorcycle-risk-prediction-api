from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_rejects_empty_payload():
    response = client.post("/api/v1/risk/predict-risk", json={})
    assert response.status_code == 422
