"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from ..server import create_app
from ..config import ServerConfig


@pytest.fixture
def client():
    """Create a test client."""
    config = ServerConfig(
        host="localhost",
        port=8000,
        api_key="test-key",
        log_level="INFO"
    )
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
                    "seed": 42
                }
            }
        }
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
