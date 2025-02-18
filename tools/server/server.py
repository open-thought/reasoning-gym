"""FastAPI server implementation for Reasoning Gym."""

import logging
from fastapi import FastAPI
from .config import ServerConfig
from .middleware import APIKeyMiddleware
from .registry import ExperimentRegistry


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
    
    return app
