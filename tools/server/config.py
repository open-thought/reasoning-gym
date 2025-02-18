"""Server configuration using Pydantic settings management."""

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Configuration settings for the Reasoning Gym server."""

    host: str = Field(default="localhost", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    api_key: str = Field(
        default=..., description="API key for authentication", json_schema_extra={"env": "REASONING_GYM_API_KEY"}
    )
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = ConfigDict(env_prefix="REASONING_GYM_")
