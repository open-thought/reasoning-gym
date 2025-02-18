"""HTTP client for interacting with the Reasoning Gym server."""

import os
from typing import Any, Dict, Optional

import httpx
from rich.console import Console

console = Console()

DEFAULT_SERVER = "http://localhost:8000"
API_KEY = os.getenv("REASONING_GYM_API_KEY", "default-key")


class RGClient:
    """Client for interacting with Reasoning Gym server."""

    def __init__(self, base_url: str = DEFAULT_SERVER, api_key: str = API_KEY):
        """Initialize client with server URL and API key."""
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}

    def _url(self, path: str) -> str:
        """Construct full URL for given path."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def check_health(self) -> bool:
        """Check server health status."""
        try:
            response = httpx.get(self._url("/health"), headers=self.headers)
            response.raise_for_status()
            return response.json()["status"] == "healthy"
        except Exception:
            return False

    def list_experiments(self) -> list[str]:
        """List all registered experiments."""
        response = httpx.get(self._url("/experiments"), headers=self.headers)
        response.raise_for_status()
        return response.json()["experiments"]

    def create_experiment(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new experiment."""
        response = httpx.post(
            self._url("/experiments"),
            headers=self.headers,
            json={"name": name, **config},
        )
        response.raise_for_status()
        return response.json()

    def delete_experiment(self, name: str) -> None:
        """Delete an experiment."""
        response = httpx.delete(
            self._url(f"/experiments/{name}"),
            headers=self.headers,
        )
        response.raise_for_status()

    def get_experiment_config(self, name: str) -> Dict[str, Any]:
        """Get experiment configuration."""
        response = httpx.get(
            self._url(f"/experiments/{name}/composite"),
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def update_dataset_config(self, experiment: str, dataset: str, config: Dict[str, Any]) -> None:
        """Update dataset configuration."""
        response = httpx.post(
            self._url(f"/experiments/{experiment}/composite/{dataset}"),
            headers=self.headers,
            json={"config": config},
        )
        response.raise_for_status()
