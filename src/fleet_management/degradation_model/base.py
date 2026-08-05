from typing import Protocol

import numpy as np


class AccumulatedDegradationModel(Protocol):
    """Mathematical interface for accumulated-degradation models."""

    name: str

    def increment_parameter(
        self,
        expected_damage: np.ndarray,
    ) -> np.ndarray:
        """Convert expected mission damage into model parameters."""
        ...

    def expected_damage(
        self,
        state_parameter: np.ndarray,
    ) -> np.ndarray:
        """Return expected damage represented by the state."""
        ...

    def tail_probability(
        self,
        state_parameter: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Return P(D > threshold)."""
        ...