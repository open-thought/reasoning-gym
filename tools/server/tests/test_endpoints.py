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
                    "seed": 42
                }
            }
        }
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
    update_data = {
        "config": {
            "min_terms": 3,
            "max_terms": 5
        }
    }
    response = client.post(
        "/experiments/test_exp/composite/chain_sum",
        json=update_data,
        headers=headers
    )
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
    response = client.post(
        "/experiments/test_exp/composite/nonexistent",
        json=update_data,
        headers=headers
    )
    assert response.status_code == 404
