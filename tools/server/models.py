"""Pydantic models for API request/response data."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    """Request model for creating a new experiment."""

    name: str = Field(..., description="Unique name for the experiment")
    size: int = Field(500, description="Size of the dataset")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    datasets: Dict[str, Dict[str, Any]] = Field(..., description="Dictionary of datasets configurations")


class ExperimentResponse(BaseModel):
    """Response model for experiment operations."""

    name: str = Field(..., description="Name of the experiment")
    size: int = Field(..., description="Size of the dataset")
    seed: Optional[int] = Field(None, description="Random seed used")
    datasets: Dict[str, Dict[str, Any]] = Field(..., description="Current dataset configurations")


class ExperimentList(BaseModel):
    """Response model for listing experiments."""

    experiments: list[str] = Field(default_factory=list, description="List of registered experiment names")


class DatasetConfigUpdate(BaseModel):
    """Request model for updating dataset configuration."""

    config: Dict[str, Any] = Field(..., description="Configuration parameters to update")


class ErrorResponse(BaseModel):
    """Response model for error conditions."""

    detail: str = Field(..., description="Error message")


@dataclass
class BatchEntry:
    """Single entry in a batch"""

    question: str
    entry_id: str  # Format: "{version}.{index}"
    metadata: Dict[str, Any]


@dataclass
class BatchResponse:
    """Response containing a batch of entries"""

    entries: List[BatchEntry]


@dataclass
class ScoringRequest:
    """Request for scoring model outputs"""

    scores: List[Tuple[str, str]]  # List of (entry_id, answer) pairs
