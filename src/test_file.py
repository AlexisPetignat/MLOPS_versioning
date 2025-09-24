# test_predict.py
import pytest
from fastapi.testclient import TestClient
from server import (
    app,
)  # import your FastAPI app (adjust if your file is named differently)

client = TestClient(app)

# ------------------
# Fixtures
# ------------------


@pytest.fixture
def valid_fish():
    """A valid fish input with typical values."""
    return {
        "Height": 5.5,
        "Width": 3.2,
        "Length1": 10.0,
        "Length2": 11.0,
        "Length3": 12.0,
    }


@pytest.fixture
def tiny_fish():
    """Edge case: very small values (approaching zero)."""
    return {
        "Height": 0.01,
        "Width": 0.01,
        "Length1": 0.01,
        "Length2": 0.01,
        "Length3": 0.01,
    }


@pytest.fixture
def huge_fish():
    """Edge case: very large values (potential overflow)."""
    return {
        "Height": 1e6,
        "Width": 1e6,
        "Length1": 1e6,
        "Length2": 1e6,
        "Length3": 1e6,
    }


@pytest.fixture
def new_params():
    """New parameters for model"""
    return {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 2,
        "random_state": 69,
    }


# ------------------
# Tests
# ------------------


def test_predict_valid(valid_fish):
    response = client.post("/predict", json=valid_fish)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1  # model.predict should return 1 value


def test_predict_tiny(tiny_fish):
    response = client.post("/predict", json=tiny_fish)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)


def test_predict_huge(huge_fish):
    response = client.post("/predict", json=huge_fish)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)


def test_predict_missing_field(valid_fish):
    # Remove one required field -> should fail validation
    fish = valid_fish.copy()
    fish.pop("Height")
    response = client.post("/predict", json=fish)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)


def test_predict_invalid_type(valid_fish):
    # Wrong type for one field
    fish = valid_fish.copy()
    fish["Width"] = "not_a_number"
    response = client.post("/predict", json=fish)
    assert response.status_code == 422


def test_change_model(new_params, valid_fish):
    # Fetch prediction, then change, then predict, then compare
    response1 = client.post("/predict", json=valid_fish)
    client.post("/update-model", json=new_params)
    response2 = client.post("/predict", json=valid_fish)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() != response2.json()
