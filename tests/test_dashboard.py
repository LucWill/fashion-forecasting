from starlette.testclient import TestClient
from fashion_pipeline.report.dashboard import app
from unittest.mock import patch

client = TestClient(app)


def test_home_route_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "StyleSense Recommendation Predictor" in response.text


@patch("fashion_pipeline.report.dashboard.load_model")
def test_predict_route_returns_probability(mock_load_model):
    mock_model = mock_load_model.return_value
    mock_model.predict_proba.return_value = [[0.1, 0.9]]  # fake probability

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
