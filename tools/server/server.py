"""FastAPI server implementation for Reasoning Gym."""

import logging
from fastapi import FastAPI, HTTPException
from .config import ServerConfig
from .middleware import APIKeyMiddleware
from .registry import ExperimentRegistry
from .models import ExperimentCreate, ExperimentResponse, ExperimentList
from reasoning_gym.composite import CompositeConfig, DatasetSpec


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
            dataset_specs.append(
                DatasetSpec(
                    name=name,
                    weight=spec.get("weight", 1.0),
                    config=spec.get("config", {})
                )
            )
        
        config = CompositeConfig(
            size=experiment.size,
            seed=experiment.seed,
            datasets=dataset_specs
        )
        
        try:
            registry.register_experiment(experiment.name, config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        return ExperimentResponse(
            name=experiment.name,
            size=experiment.size,
            seed=experiment.seed,
            datasets=experiment.datasets
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
    
    return app
