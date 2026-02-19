"""Strategy interfaces for algorithm component injection."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .strategy_models import (
    ComponentAssignmentRequest,
    CovarianceRequest,
    InitializationRequest,
    LossRequest,
    NMFUpdateRequest,
    NMFUpdateResult,
    NormalizationRequest,
    OnlineFrameRequest,
    PermutationRequest,
    ReconstructionRequest,
    ReconstructionResult,
    SourceModelRequest,
    SourceModelResult,
)


class InitializationStrategy(ABC):
    """Initializes algorithm parameters/states."""

    @abstractmethod
    def initialize(self, request: InitializationRequest) -> dict[str, np.ndarray]:
        """Return initialized array parameters."""


class SpatialUpdateStrategy(ABC):
    """Updates spatial parameters such as demixing filters."""

    def row_groups(self, n_sources: int) -> list[int | np.ndarray]:
        """Return row-index groups for one spatial update pass."""
        return list(range(n_sources))

    @abstractmethod
    def update(
        self,
        covariance: np.ndarray,
        demix_filter: np.ndarray,
        *,
        row_idx: int | np.ndarray,
    ) -> np.ndarray:
        """Return updated demixing filter."""


class CovarianceUpdateStrategy(ABC):
    """Updates weighted covariance statistics."""

    @abstractmethod
    def update(self, request: CovarianceRequest) -> np.ndarray:
        """Return updated covariance tensor."""


class SourceModelStrategy(ABC):
    """Updates source model parameters (variance, NMF factors, etc.)."""

    @abstractmethod
    def update(self, request: SourceModelRequest) -> SourceModelResult:
        """Return updated source model tensors."""


class NormalizationStrategy(ABC):
    """Applies normalization or scale restoration."""

    @abstractmethod
    def apply(self, request: NormalizationRequest) -> np.ndarray:
        """Return normalized estimate."""


class PermutationStrategy(ABC):
    """Solves source permutation ambiguity."""

    @abstractmethod
    def solve(self, request: PermutationRequest) -> np.ndarray:
        """Return permutation indices."""


class LossStrategy(ABC):
    """Computes optimization loss values."""

    @abstractmethod
    def compute(self, request: LossRequest) -> np.ndarray | float:
        """Return current loss value."""


class ReconstructionStrategy(ABC):
    """Constructs separated outputs from intermediate representations."""

    @abstractmethod
    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        """Return separated signal and optional mask."""


class OnlineNMFUpdateStrategy(ABC):
    """Updates online NMF state (activation, dictionary, sufficient statistics)."""

    @abstractmethod
    def update(self, request: NMFUpdateRequest) -> NMFUpdateResult:
        """Return updated online NMF state for one frame."""


class ComponentAssignmentStrategy(ABC):
    """Resolves NMF component-to-source assignments."""

    @abstractmethod
    def resolve(self, request: ComponentAssignmentRequest) -> np.ndarray:
        """Return assignment indices of shape ``(n_components,)``."""


class OnlineFrameOptionStrategy(ABC):
    """Normalizes optional per-frame runtime options for streaming separators."""

    @abstractmethod
    def resolve(self, request: OnlineFrameRequest | None) -> OnlineFrameRequest:
        """Return normalized frame options."""


SpatialStrategy = SpatialUpdateStrategy
CovarianceStrategy = CovarianceUpdateStrategy
SourceStrategy = SourceModelStrategy
NMFStrategy = OnlineNMFUpdateStrategy
AssignmentStrategy = ComponentAssignmentStrategy
