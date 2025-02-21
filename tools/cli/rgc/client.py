"""HTTP client for interacting with the Reasoning Gym server."""

import os
from typing import List, Optional

import httpx
from rich.console import Console

from tools.server.models import (
    AnswerItem,
    BatchResponse,
    DatasetConfigUpdate,
    ExperimentCreate,
    ExperimentList,
    ExperimentResponse,
    ScoringRequest,
    ScoringResponse,
)

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

    def list_experiments(self) -> ExperimentList:
        """List all registered experiments."""
        response = httpx.get(self._url("/experiments"), headers=self.headers)
        response.raise_for_status()
        return ExperimentList.model_validate(response.json())

    def create_experiment(self, name: str, config: ExperimentCreate) -> ExperimentResponse:
        """Create a new experiment."""
        response = httpx.post(
            self._url("/experiments"),
            headers=self.headers,
            json=config.model_dump(),
        )
        response.raise_for_status()
        return ExperimentResponse.model_validate(response.json())

    def delete_experiment(self, name: str) -> None:
        """Delete an experiment."""
        response = httpx.delete(
            self._url(f"/experiments/{name}"),
            headers=self.headers,
        )
        response.raise_for_status()

    def get_experiment_config(self, name: str) -> ExperimentResponse:
        """Get experiment configuration."""
        response = httpx.get(
            self._url(f"/experiments/{name}/composite"),
            headers=self.headers,
        )
        response.raise_for_status()
        return ExperimentResponse.model_validate(response.json())

    def update_dataset_config(self, experiment: str, dataset: str, config: DatasetConfigUpdate) -> None:
        """Update dataset configuration."""
        response = httpx.post(
            self._url(f"/experiments/{experiment}/composite/{dataset}"),
            headers=self.headers,
            json=config.model_dump(),
        )
        response.raise_for_status()

    def get_batch(self, experiment: str, base_index: int, batch_size: int) -> BatchResponse:
        """Get a batch of entries from an experiment.

        Args:
            experiment: Name of the experiment
            base_index: Starting index for the batch
            batch_size: Number of entries to retrieve

        Returns:
            BatchResponse containing entries with questions and metadata
        """
        response = httpx.get(
            self._url(f"/experiments/{experiment}/batch"),
            headers=self.headers,
            params={"base_index": base_index, "batch_size": batch_size},
        )
        response.raise_for_status()
        return BatchResponse.model_validate(response.json())

    def score_outputs(self, experiment: str, entry_answers: list[AnswerItem]) -> ScoringResponse:
        """Score a batch of answers.

        Args:
            experiment: Name of the experiment
            entry_answers: List of AnswerItems with entry_ids and answers to score

        Returns:
            ScoringResponse containing scores and entry_ids
        """
        request = ScoringRequest(answers=entry_answers)
        response = httpx.post(
            self._url(f"/experiments/{experiment}/score"),
            headers=self.headers,
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ScoringResponse.model_validate(response.json())
