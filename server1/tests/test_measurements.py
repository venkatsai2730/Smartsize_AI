import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_images():
    with open("tests/sample_front.jpg", "rb") as front, open("tests/sample_side.jpg", "rb") as side:
        response = client.post(
            "/api/v1/upload",
            files={"front_image": front, "side_image": side},
            data={"height": 170.0},
        )
    assert response.status_code == 200
    assert "measurement_id" in response.json()