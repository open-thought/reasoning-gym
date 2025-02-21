"""Version manager for tracking dataset versions."""

from typing import Any, Optional

from .dataset import ProceduralDataset


class DatasetVersionManager:
    """Manages versioned ProceduralDataset instances and their configurations."""

    def __init__(self):
        """Initialize the version manager."""
        self.current_version = 0
        # version_id -> (dataset_name, dataset_instance)
        self.datasets: dict[int, tuple[str, ProceduralDataset]] = {}

    def register_dataset(self, name: str, dataset: ProceduralDataset) -> int:
        """
        Register a new dataset version.

        Args:
            name: Name/identifier of the dataset type
            dataset: Instance of ProceduralDataset

        Returns:
            version_id: Unique identifier for this dataset version
        """
        self.current_version += 1
        self.datasets[self.current_version] = (name, dataset)
        return self.current_version

    def get_dataset(self, version_id: int) -> Optional[tuple[str, ProceduralDataset]]:
        """
        Retrieve a dataset by its version ID.

        Args:
            version_id: The version identifier

        Returns:
            Tuple of (dataset_name, dataset_instance) if found, None otherwise
        """
        return self.datasets.get(version_id)

    def get_entry(self, version_id: int, index: int) -> dict[str, Any]:
        """
        Get a specific entry from a versioned dataset.

        Args:
            version_id: The version identifier
            index: Index of the entry to retrieve

        Returns:
            The dataset entry

        Raises:
            KeyError: If version_id is not found
        """
        if version_id not in self.datasets:
            raise KeyError(f"Dataset version {version_id} not found")

        _, dataset = self.datasets[version_id]
        return dataset[index]

    def cleanup_old_versions(self, keep_latest: int = 10):
        """
        Remove old dataset versions to free memory.

        Args:
            keep_latest: Number of most recent versions to keep
        """
        if len(self.datasets) <= keep_latest:
            return

        versions_to_remove = sorted(self.datasets.keys())[:-keep_latest]
        for version in versions_to_remove:
            del self.datasets[version]
