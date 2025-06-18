import pytest
from fastapi.testclient import TestClient
from fashion_pipeline.report.dashboard_htmx import app



client = TestClient(app)


def test_home_route_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "StyleSense Recommendation Predictor" in response.text


def test_predict_route_returns_probability():
    response = client.post("/predict", data={
        "title": "Love this product",
        "review_text": "This dress was fantastic. I would buy it again!",
        "age": "32",
        "division": "General",
        "department": "Dresses",
        "class_name": "Dresses"
    })
    assert response.status_code == 200
    assert "Recommendation Probability" in response.text
