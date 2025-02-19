"""FastAPI server implementation for Reasoning Gym."""

import logging

from fastapi import FastAPI, HTTPException

from reasoning_gym.coaching.registry import ExperimentRegistry
from reasoning_gym.composite import CompositeConfig, DatasetSpec

from .config import ServerConfig
from .middleware import APIKeyMiddleware
from .models import (
    BatchEntry,
    BatchResponse,
    DatasetConfigUpdate,
    ExperimentCreate,
    ExperimentList,
    ExperimentResponse,
    ScoringRequest,
    ScoringResponse,
)


def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    # Configure logging
    logging.basicConfig(level=config.log_level)
    logger = logging.getLogger(__name__)

    # Create FastAPI app
    app = FastAPI(title="Reasoning Gym Server")

    # Add middleware
    app.add_middleware(APIKeyMiddleware, api_key=config.api_key)

    # Initialize registry
    registry = ExperimentRegistry()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/experiments", response_model=ExperimentResponse)
    async def create_experiment(experiment: ExperimentCreate):
        """Create a new experiment."""
        # Convert dict format to DatasetSpec list
        dataset_specs = []
        for name, spec in experiment.datasets.items():
            dataset_specs.append(DatasetSpec(name=name, weight=spec.get("weight", 1.0), config=spec.get("config", {})))

        config = CompositeConfig(size=experiment.size, seed=experiment.seed, datasets=dataset_specs)

        try:
            registry.register_experiment(experiment.name, config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return ExperimentResponse(
            name=experiment.name, size=experiment.size, seed=experiment.seed, datasets=experiment.datasets
        )

    @app.get("/experiments", response_model=ExperimentList)
    async def list_experiments():
        """List all registered experiments."""
        return ExperimentList(experiments=registry.list_experiments())

    @app.delete("/experiments/{name}")
    async def delete_experiment(name: str):
        """Delete an experiment."""
        if not registry.remove_experiment(name):
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
        return {"status": "deleted"}

    @app.get("/experiments/{name}/batch", response_model=BatchResponse)
    async def generate_batch(name: str, base_index: int, batch_size: int):
        """Generate a batch of raw entries"""
        # Validate parameters
        if base_index < 0:
            raise HTTPException(status_code=400, detail="base_index must be non-negative")
        if batch_size <= 0:
            raise HTTPException(status_code=400, detail="batch_size must be positive")

        experiment = registry.get_experiment(name)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")

        try:
            entries = []
            for i in range(base_index, base_index + batch_size):
                entry = experiment.dataset[i]

                # Create BatchEntry with minimal required data
                batch_entry = BatchEntry(
                    question=entry["question"],
                    entry_id=f"{entry['metadata']['version_id']}.{i}",
                    metadata=entry["metadata"],
                )
                entries.append(batch_entry)

            return BatchResponse(entries=entries)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/experiments/{name}/score", response_model=ScoringResponse)
    async def score_outputs(name: str, request: ScoringRequest):
        """Score extracted answers"""
        experiment = registry.get_experiment(name)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")

        try:
            scores = []
            entry_ids = []
            for item in request.answers:
                score = experiment.dataset.score_answer_with_id(item.answer, item.entry_id)
                scores.append(score)
                entry_ids.append(item.entry_id)

            return ScoringResponse(scores=scores, entry_ids=entry_ids)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/experiments/{name}/composite", response_model=ExperimentResponse)
    async def get_composite_config(name: str):
        """Get composite configuration for an experiment."""
        experiment = registry.get_experiment(name)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")

        # Convert internal config to API response format
        datasets = {}
        for ds_spec in experiment.config.datasets:
            dataset = experiment.dataset.datasets[ds_spec.name]
            datasets[ds_spec.name] = {
                "weight": ds_spec.weight,
                "config": vars(dataset.config),  # Get current config from dataset instance
            }

        return ExperimentResponse(
            name=name, size=experiment.config.size, seed=experiment.config.seed, datasets=datasets
        )

    @app.post("/experiments/{name}/composite/{dataset_name}")
    async def update_dataset_config(name: str, dataset_name: str, config_update: DatasetConfigUpdate):
        """Update configuration for a specific dataset in the composite."""
        experiment = registry.get_experiment(name)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")

        try:
            experiment.dataset.update_dataset_config(dataset_name, config_update.config)
            return {"status": "updated"}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found in experiment")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


async def app(scope, receive, send):
    """ASGI application that lazily creates the FastAPI app."""
    if not hasattr(app, "server_app"):
        app.server_app = create_app(ServerConfig())
    await app.server_app(scope, receive, send)
