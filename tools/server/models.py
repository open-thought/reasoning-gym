"""Pydantic models for API request/response data."""

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

    experiments: List[str] = Field(default_factory=list, description="List of registered experiment names")


class DatasetConfigUpdate(BaseModel):
    """Request model for updating dataset configuration."""

    config: Dict[str, Any] = Field(..., description="Configuration parameters to update")


class ErrorResponse(BaseModel):
    """Response model for error conditions."""

    detail: str = Field(..., description="Error message")


class BatchEntry(BaseModel):
    """Single entry in a batch"""

    question: str = Field(..., description="The question text")
    entry_id: str = Field(..., description="Unique identifier in format '{version}.{index}'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the entry")


class BatchResponse(BaseModel):
    """Response containing a batch of entries"""

    entries: List[BatchEntry] = Field(..., description="List of batch entries")


class AnswerItem(BaseModel):
    """Single score item containing entry_id and answer"""

    entry_id: str = Field(..., description="Entry identifier to score")
    answer: str = Field(..., description="Answer to evaluate")


class ScoringRequest(BaseModel):
    """Request for scoring model outputs"""

    answers: List[AnswerItem] = Field(..., description="List of entries to score")


class ScoringResponse(BaseModel):
    """Response containing scores for answers"""

    scores: List[float] = Field(..., description="List of scores in same order as request")
    entry_ids: List[str] = Field(..., description="List of entry_ids in same order as request")
