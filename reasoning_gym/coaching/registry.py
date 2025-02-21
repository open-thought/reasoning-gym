"""Registry for managing active experiments."""

from typing import Optional

from ..composite import CompositeConfig
from .experiment import Experiment


class ExperimentRegistry:
    """Singleton registry for managing active experiments."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._experiments = {}
        return cls._instance

    def register_experiment(self, name: str, config: CompositeConfig) -> None:
        """Register a new experiment with the given name and configuration."""
        self._experiments[name] = Experiment.create(name, config)

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        return self._experiments.get(name)

    def list_experiments(self) -> list[str]:
        """List all registered experiment names."""
        return list(self._experiments.keys())

    def remove_experiment(self, name: str) -> bool:
        """Remove an experiment by name. Returns True if removed, False if not found."""
        return bool(self._experiments.pop(name, None))
