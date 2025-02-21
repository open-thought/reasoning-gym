"""Tests for server configuration."""

import os

import pytest

from ..config import ServerConfig


def test_default_config():
    """Test default configuration values."""
    os.environ["REASONING_GYM_API_KEY"] = "test-key"
    config = ServerConfig()

    assert config.host == "localhost"
    assert config.port == 8000
    assert config.api_key == "test-key"
    assert config.log_level == "INFO"


def test_missing_api_key():
    """Test that missing API key raises an error."""
    if "REASONING_GYM_API_KEY" in os.environ:
        del os.environ["REASONING_GYM_API_KEY"]

    with pytest.raises(ValueError):
        ServerConfig()
