"""Concrete strategy implementations for separator components."""

from .injection import (
    AuxSourceStrategy,
    BatchCovarianceStrategy,
    default_strategy_registry,
    EMACovarianceStrategy,
    GaussSourceStrategy,
    ILRMANMFSourceStrategy,
    IP1SpatialStrategy,
    IP2SpatialStrategy,
    ISS1SpatialStrategy,
    LaplaceSourceStrategy,
    ModuloAssignmentStrategy,
    MultiplicativeNMFStrategy,
    RuleSpatialStrategy,
    StrategyRegistry,
    StrategyRegistryError,
    WeightedCovarianceStrategy,
)
from .normalization import IdentityNormalization, ProjectionBackNormalization
from .permutation import (
    BssEvalPermutationStrategy,
    IdentityPermutationStrategy,
    ScoreMatrixPermutationStrategy,
)
from .reconstruction import (
    DemixReconstructionStrategy,
    ProjectionBackDemixReconstructionStrategy,
    RatioMaskReconstructionStrategy,
)

__all__ = [
    "StrategyRegistry",
    "StrategyRegistryError",
    "default_strategy_registry",
    "RuleSpatialStrategy",
    "IP1SpatialStrategy",
    "IP2SpatialStrategy",
    "ISS1SpatialStrategy",
    "AuxSourceStrategy",
    "GaussSourceStrategy",
    "LaplaceSourceStrategy",
    "ILRMANMFSourceStrategy",
    "WeightedCovarianceStrategy",
    "BatchCovarianceStrategy",
    "EMACovarianceStrategy",
    "MultiplicativeNMFStrategy",
    "ModuloAssignmentStrategy",
    "IdentityNormalization",
    "ProjectionBackNormalization",
    "IdentityPermutationStrategy",
    "ScoreMatrixPermutationStrategy",
    "BssEvalPermutationStrategy",
    "DemixReconstructionStrategy",
    "ProjectionBackDemixReconstructionStrategy",
    "RatioMaskReconstructionStrategy",
]
