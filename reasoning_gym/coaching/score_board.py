"""Coaching module for difficulty adjustment and score tracking"""

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional, Union

from ..dataset import ProceduralDataset


@dataclass
class ScoreStats:
    """Container for score statistics with mean, std, min, max"""

    scores: OrderedDict[tuple[tuple[str, Any], ...], tuple[int, float, float, float, float]]

    def __str__(self) -> str:
        """Create a formatted report of the statistics

        Returns:
            Multi-line string with statistics for each group
        """
        if not self.scores:
            return "No scores recorded"

        lines = []

        for key, values in self.scores.items():
            params = ", ".join(f"{k}={v}" for k, v in key)
            lines.append(
                f"({params}): n={values[0]}, μ={values[1]:.3f}, σ={values[2]:.3f}, min={values[3]:.3f}, max={values[4]:.3f}"
            )

        return "\n".join(lines)


@dataclass
class GroupedScores:
    """Container for grouped scores with total count"""

    scores: OrderedDict[tuple[tuple[str, Any], ...], list[float]]
    total_scores: int

    def __str__(self) -> str:
        """Create a formatted report of the scores

        Returns:
            Multi-line string with score information for each difficulty group
        """
        if not self.scores:
            return "No scores recorded"

        lines = []
        lines.append(f"Total scores: {self.total_scores}")
        lines.append("")

        for key, values in self.scores.items():
            # Format the parameter combinations
            params = ", ".join(f"{k}={v}" for k, v in key)
            stats = (
                len(values),
                mean(values) if values else 0.0,
                stdev(values) if len(values) > 1 else 0.0,
                min(values) if values else 0.0,
                max(values) if values else 0.0,
            )
            lines.append(
                f"({params}): n={stats[0]}, μ={stats[1]:.3f}, σ={stats[2]:.3f}, min={stats[3]:.3f}, max={stats[4]:.3f}"
            )
            # Format score list, showing only last 100 if more
            score_strs = [f"{x:.2f}" for x in values[-100:]]
            if len(values) > 100:
                score_strs.insert(0, "..")
            lines.append(f"  Values: {', '.join(score_strs)}")

        return "\n".join(lines)

    def stats(self, ignore_empty: bool = True) -> ScoreStats:
        """Calculate statistics for each group of scores

        Args:
            ignore_empty: If True, skip empty score lists
                         If False, use NaN values for empty lists

        Returns:
            ScoreStats object containing statistics for each group
        """
        result = OrderedDict()

        for key, values in self.scores.items():
            if not values and ignore_empty:
                continue

            if not values:
                # Empty list and not ignoring - use NaN
                result[key] = (0, math.nan, math.nan, math.nan, math.nan)
            else:
                # Calculate stats as tuple
                result[key] = (
                    len(values),
                    mean(values),
                    stdev(values) if len(values) > 1 else 0.0,
                    min(values),
                    max(values),
                )

        return ScoreStats(scores=result)


@dataclass
class ScoreBoard:
    """Tracks scores and metadata for coaching sessions"""

    scores: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    conversations: dict[str, list[Optional[list[dict]]]] = field(default_factory=dict)

    def add_score(
        self, dataset_name: str, score: float, metadata: dict[str, Any], conversation: Optional[list[dict]] = None
    ) -> None:
        """Add a new score entry with associated metadata and optional conversation

        Args:
            score: The score achieved (typically 0.0-1.0)
            metadata: Dictionary of metadata about the task/attempt
            conversation: Optional list of conversation turns as dicts
        """
        if dataset_name not in self.scores:
            self.scores[dataset_name] = []
            self.metadata[dataset_name] = []
            self.conversations[dataset_name] = []
        self.scores[dataset_name].append(score)
        self.metadata[dataset_name].append(metadata)
        self.conversations[dataset_name].append(conversation)

    def clear(self, dataset_name: str) -> None:
        """Clear all stored scores, metadata and conversations"""
        self.scores[dataset_name] = []
        self.metadata[dataset_name] = []
        self.conversations[dataset_name] = []

    def __len__(self) -> int:
        """Return the number of stored scores"""
        return len(self.scores)

    def _metadata_to_key(self, metadata: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        """Convert metadata dict to tuple of key-value pairs, sorted by key

        If source_dataset and source_index are present in metadata, they will be
        placed first in the tuple as ("source", dataset) and ("idx", index).
        """
        # Start with empty list
        key_items = [("source", metadata["source_dataset"])]

        # Add difficulty parameters or other metadata
        if "difficulty" in metadata:
            # Use only difficulty parameters
            items = metadata["difficulty"].items()
        else:
            # Use all metadata except source info
            items = ((k, v) for k, v in metadata.items() if k not in ("source_dataset"))

        # Add remaining items in sorted order
        key_items.extend(sorted((str(k), v) for k, v in items))

        return tuple(key_items)

    def aggregate(self, last_n: Optional[int] = None) -> dict[str, GroupedScores]:
        """Aggregate scores by dataset name and then by difficulty parameters

        Args:
            last_n: Optional number of most recent entries to consider
                If None, use all entries

        Returns:
            Dictionary mapping dataset names to their respective GroupedScores objects
            Each GroupedScores contains scores grouped by difficulty parameters for that dataset
        """
        if not self.scores:
            return {}

        # Create a nested structure: dataset -> parameter groups -> scores
        result = {}

        # Process each dataset
        for dataset_name, dataset_scores in self.scores.items():
            # Determine start index for this dataset
            dataset_len = len(dataset_scores)
            start_idx = max(0, dataset_len - last_n) if last_n is not None else 0

            # Create OrderedDict for this dataset's parameter groupings
            dataset_groups = OrderedDict()

            # Process scores for this dataset
            for i in range(start_idx, dataset_len):
                # Get metadata for this score
                metadata = self.metadata[dataset_name][i]
                params = self._metadata_to_key(metadata)

                if params not in dataset_groups:
                    dataset_groups[params] = []

                dataset_groups[params].append(dataset_scores[i])

            # Create a GroupedScores object for this dataset
            total_scores = sum(len(scores) for scores in dataset_groups.values())
            result[dataset_name] = GroupedScores(scores=dataset_groups, total_scores=total_scores)

        return result
