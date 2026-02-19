"""Core abstractions for reusable separator design.

This package provides shared building blocks used by both batch and online
separation algorithms:

- Typed input/output containers.
- Base separator classes.
- Strategy interfaces.
- Lightweight algorithm registry.
- Backend abstraction for NumPy/Torch interoperability.
"""

from .backend import Backend, NumpyBackend, TorchBackend
from .base import BaseIterativeSeparator, BaseSeparator, BaseStreamingSeparator
from .interfaces import (
    BatchSeparatorProtocol,
    SeparatorProtocol,
    StreamingSeparatorProtocol,
)
from .io_models import (
    BatchRequest,
    SeparationOutput,
    SeparatorState,
    StreamRequest,
    StreamingSeparatorState,
)
from .registry import RegistryError, SeparatorRegistry
from .strategy_models import (
    ComponentAssignmentRequest,
    CovarianceRequest,
    InitializationRequest,
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
from .strategies import (
    AssignmentStrategy,
    CovarianceStrategy,
    InitializationStrategy,
    ComponentAssignmentStrategy,
    CovarianceUpdateStrategy,
    LossStrategy,
    NormalizationStrategy,
    OnlineFrameOptionStrategy,
    OnlineNMFUpdateStrategy,
    PermutationStrategy,
    ReconstructionStrategy,
    SourceModelStrategy,
    SourceStrategy,
    SpatialStrategy,
    SpatialUpdateStrategy,
    NMFStrategy,
)

__all__ = [
    "Backend",
    "NumpyBackend",
    "TorchBackend",
    "BaseSeparator",
    "BaseIterativeSeparator",
    "BaseStreamingSeparator",
    "SeparatorProtocol",
    "BatchSeparatorProtocol",
    "StreamingSeparatorProtocol",
    "BatchRequest",
    "StreamRequest",
    "SeparationOutput",
    "SeparatorState",
    "StreamingSeparatorState",
    "SeparatorRegistry",
    "RegistryError",
    "SourceModelRequest",
    "SourceModelResult",
    "CovarianceRequest",
    "InitializationRequest",
    "NMFUpdateRequest",
    "NMFUpdateResult",
    "ComponentAssignmentRequest",
    "OnlineFrameRequest",
    "NormalizationRequest",
    "PermutationRequest",
    "ReconstructionRequest",
    "ReconstructionResult",
    "SpatialUpdateStrategy",
    "SpatialStrategy",
    "CovarianceUpdateStrategy",
    "CovarianceStrategy",
    "SourceModelStrategy",
    "SourceStrategy",
    "LossStrategy",
    "OnlineNMFUpdateStrategy",
    "NMFStrategy",
    "ComponentAssignmentStrategy",
    "AssignmentStrategy",
    "OnlineFrameOptionStrategy",
    "NormalizationStrategy",
    "PermutationStrategy",
    "ReconstructionStrategy",
    "InitializationStrategy",
]
