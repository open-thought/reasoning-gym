"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from ..config import ServerConfig
from ..server import create_app


@pytest.fixture
def client():
    """Create a test client."""
    config = ServerConfig(host="localhost", port=8000, api_key="test-key", log_level="INFO")
    app = create_app(config)
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_experiment_endpoints(client):
    """Test experiment management endpoints."""
    # Set API key
    headers = {"X-API-Key": "test-key"}

    # Create experiment
    create_data = {
        "name": "test_exp",
        "size": 10,
        "seed": 42,
        "datasets": {
            "chain_sum": {
                "weight": 1.0,
                "config": {
                    "min_terms": 2,
                    "max_terms": 4,
                    "min_digits": 1,
                    "max_digits": 2,
                    "allow_negation": False,
                    "size": 10,
                    "seed": 42,
                },
            }
        },
    }

    response = client.post("/experiments", json=create_data, headers=headers)
    assert response.status_code == 200
    assert response.json()["name"] == "test_exp"

    # List experiments
    response = client.get("/experiments", headers=headers)
    assert response.status_code == 200
    assert "test_exp" in response.json()["experiments"]

    # Delete experiment
    response = client.delete("/experiments/test_exp", headers=headers)
    assert response.status_code == 200

    # Verify deletion
    response = client.get("/experiments", headers=headers)
    assert response.status_code == 200
    assert "test_exp" not in response.json()["experiments"]

    # Try to delete non-existent experiment
    response = client.delete("/experiments/nonexistent", headers=headers)
    assert response.status_code == 404


def test_batch_generation_endpoint(client):
    """Test batch generation endpoint."""
    headers = {"X-API-Key": "test-key"}

    # Create test experiment
    create_data = {
        "name": "test_exp",
        "size": 10,
        "seed": 42,
        "datasets": {
            "chain_sum": {
                "weight": 1.0,
                "config": {
                    "min_terms": 2,
                    "max_terms": 4,
                    "min_digits": 1,
                    "max_digits": 2,
                    "allow_negation": False,
                    "size": 10,
                    "seed": 42,
                },
            }
        },
    }

    response = client.post("/experiments", json=create_data, headers=headers)
    assert response.status_code == 200

    # Test batch generation
    response = client.get(
        "/experiments/test_exp/batch",
        params={"base_index": 0, "batch_size": 2},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    print(data)

    # Verify batch structure
    assert "entries" in data
    assert len(data["entries"]) == 2

    # Verify entry structure
    entry = data["entries"][0]
    assert "question" in entry
    assert "entry_id" in entry
    assert "metadata" in entry

    # Test error cases
    # Non-existent experiment
    response = client.get(
        "/experiments/nonexistent/batch",
        params={"base_index": 0, "batch_size": 2},
        headers=headers,
    )
    assert response.status_code == 404

    # Invalid parameters
    response = client.get(
        "/experiments/test_exp/batch",
        params={"base_index": -1, "batch_size": 2},
        headers=headers,
    )
    assert response.status_code == 400


def test_scoring_endpoint(client):
    """Test answer scoring endpoint."""
    headers = {"X-API-Key": "test-key"}

    # Create test experiment
    create_data = {
        "name": "test_exp",
        "size": 10,
        "seed": 42,
        "datasets": {
            "chain_sum": {
                "weight": 1.0,
                "config": {
                    "min_terms": 2,
                    "max_terms": 4,
                    "min_digits": 1,
                    "max_digits": 2,
                    "allow_negation": False,
                    "size": 10,
                    "seed": 42,
                },
            }
        },
    }

    response = client.post("/experiments", json=create_data, headers=headers)
    assert response.status_code == 200

    # Get a batch to get valid entry_ids
    response = client.get(
        "/experiments/test_exp/batch",
        params={"base_index": 0, "batch_size": 2},
        headers=headers,
    )
    assert response.status_code == 200
    batch = response.json()
    entry_id = batch["entries"][0]["entry_id"]

    # Test scoring with correct answer
    response = client.post(
        "/experiments/test_exp/score",
        json={"answers": [{"entry_id": entry_id, "answer": "4"}]},  # Assuming 2+2=4 is the first question
        headers=headers,
    )
    assert response.status_code == 200
    result = response.json()
    assert "scores" in result
    assert "entry_ids" in result
    assert len(result["scores"]) == 1
    assert len(result["entry_ids"]) == 1
    assert result["entry_ids"][0] == entry_id
    assert isinstance(result["scores"][0], float)
    assert 0 <= result["scores"][0] <= 1

    # Test scoring with wrong answer
    response = client.post(
        "/experiments/test_exp/score",
        json={"answers": [{"entry_id": entry_id, "answer": "wrong"}]},
        headers=headers,
    )
    assert response.status_code == 200
    result = response.json()
    assert result["scores"][0] < 1.0
    assert result["entry_ids"][0] == entry_id

    # Test error cases
    # Invalid entry_id format
    response = client.post(
        "/experiments/test_exp/score",
        json={"answers": [{"entry_id": "invalid_id", "answer": "4"}]},
        headers=headers,
    )
    assert response.status_code == 400

    # Non-existent experiment
    response = client.post(
        "/experiments/nonexistent/score",
        json={"answers": [{"entry_id": entry_id, "answer": "4"}]},
        headers=headers,
    )
    assert response.status_code == 404


def test_composite_config_endpoints(client):
    """Test composite configuration endpoints."""
    headers = {"X-API-Key": "test-key"}

    # Create an experiment first
    create_data = {
        "name": "test_exp",
        "size": 10,
        "seed": 42,
        "datasets": {
            "chain_sum": {
                "weight": 1.0,
                "config": {
                    "min_terms": 2,
                    "max_terms": 4,
                    "min_digits": 1,
                    "max_digits": 2,
                    "allow_negation": False,
                    "size": 10,
                    "seed": 42,
                },
            }
        },
    }

    response = client.post("/experiments", json=create_data, headers=headers)
    assert response.status_code == 200

    # Get composite config
    response = client.get("/experiments/test_exp/composite", headers=headers)
    assert response.status_code == 200
    config = response.json()
    assert config["name"] == "test_exp"
    assert "chain_sum" in config["datasets"]

    # Update dataset config
    update_data = {"config": {"min_terms": 3, "max_terms": 5}}
    response = client.post("/experiments/test_exp/composite/chain_sum", json=update_data, headers=headers)
    assert response.status_code == 200

    # Verify update
    response = client.get("/experiments/test_exp/composite", headers=headers)
    assert response.status_code == 200
    config = response.json()
    assert config["datasets"]["chain_sum"]["config"]["min_terms"] == 3
    assert config["datasets"]["chain_sum"]["config"]["max_terms"] == 5

    # Test error cases
    # Non-existent experiment
    response = client.get("/experiments/nonexistent/composite", headers=headers)
    assert response.status_code == 404

    # Non-existent dataset
    response = client.post("/experiments/test_exp/composite/nonexistent", json=update_data, headers=headers)
    assert response.status_code == 404
