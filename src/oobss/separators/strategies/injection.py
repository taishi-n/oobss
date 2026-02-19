"""Injectable strategy implementations and role-aware registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from oobss.separators.core import (
    AssignmentStrategy,
    ComponentAssignmentRequest,
    CovarianceRequest,
    CovarianceStrategy,
    NMFStrategy,
    NMFUpdateRequest,
    NMFUpdateResult,
    SourceModelRequest,
    SourceModelResult,
    SourceStrategy,
    SpatialStrategy,
)
from oobss.separators.update_rules import (
    NP_EPS,
    update_covariance,
    update_source_model,
    update_spatial_model,
)


class RuleSpatialStrategy(SpatialStrategy):
    """Spatial demixing strategy backed by ``update_spatial_model``."""

    def __init__(self, method: str = "IP1") -> None:
        method_name = str(method).upper()
        if method_name not in {"IP1", "IP2", "ISS1"}:
            raise ValueError(f"Unsupported spatial strategy method: {method}")
        self.method = method_name

    def row_groups(self, n_sources: int) -> list[int | np.ndarray]:
        if self.method == "IP2":
            if n_sources < 2:
                return [0]
            return [
                np.array([idx % n_sources, (idx + 1) % n_sources], dtype=np.int64)
                for idx in range(0, n_sources * 2, 2)
            ]
        return list(range(n_sources))

    def update(
        self,
        covariance: np.ndarray,
        demix_filter: np.ndarray,
        *,
        row_idx: int | np.ndarray,
    ) -> np.ndarray:
        return update_spatial_model(
            covariance,
            demix_filter,
            row_idx=row_idx,
            method=self.method,
        )


class IP1SpatialStrategy(RuleSpatialStrategy):
    """IP1 demixing-row strategy."""

    def __init__(self) -> None:
        super().__init__("IP1")


class IP2SpatialStrategy(RuleSpatialStrategy):
    """IP2 pairwise demixing-row strategy."""

    def __init__(self) -> None:
        super().__init__("IP2")


class ISS1SpatialStrategy(RuleSpatialStrategy):
    """ISS1 demixing strategy."""

    def __init__(self) -> None:
        super().__init__("ISS1")


class AuxSourceStrategy(SourceStrategy):
    """AuxIVA source-model strategy for Gauss/Laplace contrasts."""

    def __init__(self, model: str = "Gauss", eps: float = float(NP_EPS)) -> None:
        model_name = str(model).capitalize()
        if model_name not in {"Gauss", "Laplace"}:
            raise ValueError(f"Unsupported AuxIVA source model: {model}")
        self.model = model_name
        self.eps = float(eps)

    def update(self, request: SourceModelRequest) -> SourceModelResult:
        estimated = request.estimated
        n_freq_obj = request.n_freq
        if not isinstance(estimated, np.ndarray):
            raise TypeError("estimated must be provided as np.ndarray")
        if not isinstance(n_freq_obj, int):
            raise TypeError("n_freq must be provided as int")
        n_freq = int(n_freq_obj)

        allowed = {
            "Gauss": lambda y: (np.linalg.norm(y, axis=1) ** 2) / n_freq,
            "Laplace": lambda y: 2.0 * np.linalg.norm(y, axis=1),
        }
        y_norm = allowed[self.model](estimated)
        source_model = np.maximum(self.eps, y_norm)[:, None, :]
        source_model = np.broadcast_to(source_model, estimated.shape).copy()
        return SourceModelResult(source_model=source_model)


class GaussSourceStrategy(AuxSourceStrategy):
    """AuxIVA Gauss source-model strategy."""

    def __init__(self, eps: float = float(NP_EPS)) -> None:
        super().__init__(model="Gauss", eps=eps)


class LaplaceSourceStrategy(AuxSourceStrategy):
    """AuxIVA Laplace source-model strategy."""

    def __init__(self, eps: float = float(NP_EPS)) -> None:
        super().__init__(model="Laplace", eps=eps)


class ILRMANMFSourceStrategy(SourceStrategy):
    """One-source ILRMA NMF MU strategy."""

    def __init__(self, eps: float = float(NP_EPS)) -> None:
        self.eps = float(eps)

    def update(self, request: SourceModelRequest) -> SourceModelResult:
        basis = request.basis
        activ = request.activ
        y_power = request.y_power
        if not isinstance(basis, np.ndarray):
            raise TypeError("basis must be provided as np.ndarray")
        if not isinstance(activ, np.ndarray):
            raise TypeError("activ must be provided as np.ndarray")
        if not isinstance(y_power, np.ndarray):
            raise TypeError("y_power must be provided as np.ndarray")

        basis_new, activ_new = update_source_model(
            y_power.T, basis, activ, eps=self.eps
        )
        source_model = activ_new @ basis_new.T
        return SourceModelResult(
            source_model=source_model,
            basis=basis_new,
            activ=activ_new,
        )


class WeightedCovarianceStrategy(CovarianceStrategy):
    """Weighted covariance strategy for batch/online settings."""

    def __init__(self, alpha: float | None = None) -> None:
        self.alpha = None if alpha is None else float(alpha)

    def update(self, request: CovarianceRequest) -> np.ndarray:
        alpha = self.alpha if request.alpha is None else float(request.alpha)
        return update_covariance(
            request.observed,
            request.source_model,
            prev_cov=request.prev_cov,
            alpha=alpha,
        )


class BatchCovarianceStrategy(WeightedCovarianceStrategy):
    """Batch covariance strategy without smoothing."""

    def __init__(self) -> None:
        super().__init__(alpha=None)


class EMACovarianceStrategy(WeightedCovarianceStrategy):
    """Exponential-moving-average covariance strategy."""


class MultiplicativeNMFStrategy(NMFStrategy):
    """Online NMF strategy using multiplicative updates."""

    def update(self, request: NMFUpdateRequest) -> NMFUpdateResult:
        v = request.v
        basis = np.array(request.basis, copy=True)
        stat_a = np.array(request.stat_a, copy=True)
        stat_b = np.array(request.stat_b, copy=True)
        eps = float(request.eps)

        if request.h_prev is not None:
            h = np.array(request.h_prev, copy=True)
        else:
            h = np.ones(basis.shape[1], dtype=np.float64)

        for _ in range(request.inner_iter):
            model = basis @ h + eps
            h *= (basis.T @ (v / (model**2))) / np.maximum(
                basis.T @ (1.0 / model),
                eps,
            )

        model = basis @ h + eps
        stat_a += ((v / (model**2))[:, None] * h[None, :]) * (basis**2)
        stat_b += ((1.0 / model)[:, None]) * h[None, :]

        t = int(request.t) + 1
        batch_counter = int(request.batch_counter) + 1

        if batch_counter >= request.beta:
            rho = request.alpha ** (request.beta / max(t, 1))
            stat_a *= rho
            stat_b *= rho
            basis = np.sqrt(np.maximum(stat_a, eps) / np.maximum(stat_b, eps))

            colsum = np.maximum(basis.sum(axis=0, keepdims=True), eps)
            basis /= colsum
            stat_a /= colsum
            stat_b *= colsum
            batch_counter = 0

        return NMFUpdateResult(
            h=h,
            basis=basis,
            stat_a=stat_a,
            stat_b=stat_b,
            batch_counter=batch_counter,
            t=t,
        )


class ModuloAssignmentStrategy(AssignmentStrategy):
    """Assign components to sources via modulo fallback mapping."""

    def resolve(self, request: ComponentAssignmentRequest) -> np.ndarray:
        if request.n_sources <= 0:
            raise ValueError("n_sources must be positive")

        mapping = request.component_to_source
        if mapping is None:
            return np.arange(request.n_components, dtype=np.int64) % request.n_sources

        assignment = np.asarray(mapping, dtype=np.int64)
        if assignment.ndim != 1 or assignment.shape[0] != request.n_components:
            raise ValueError("component_to_source must be a 1-D array of length K")
        if np.any((assignment < 0) | (assignment >= request.n_sources)):
            raise ValueError("component_to_source contains out-of-range source IDs")
        return assignment


class StrategyRegistryError(RuntimeError):
    """Raised for invalid strategy-registry operations."""


@dataclass
class StrategyRegistry:
    """Role-aware name-to-factory strategy registry."""

    _factories: dict[str, dict[str, Callable[[dict[str, object]], object]]] = field(
        default_factory=dict
    )

    def register(
        self,
        role: str,
        name: str,
        factory: Callable[[dict[str, object]], object],
        *,
        overwrite: bool = False,
    ) -> None:
        role_name = str(role)
        strategy_name = str(name)
        role_map = self._factories.setdefault(role_name, {})
        if not overwrite and strategy_name in role_map:
            raise StrategyRegistryError(
                f"Strategy '{strategy_name}' for role '{role_name}' is already registered."
            )
        role_map[strategy_name] = factory

    def create(
        self,
        role: str,
        name: str,
        params: dict[str, object] | None = None,
    ) -> object:
        role_name = str(role)
        strategy_name = str(name)
        role_map = self._factories.get(role_name)
        if role_map is None or strategy_name not in role_map:
            available = ", ".join(sorted(role_map)) if role_map else "<none>"
            raise StrategyRegistryError(
                f"Unknown strategy '{strategy_name}' for role '{role_name}'. "
                f"Available: {available}"
            )
        return role_map[strategy_name]({} if params is None else dict(params))

    def available(self, role: str) -> list[str]:
        return sorted(self._factories.get(str(role), {}))


def default_strategy_registry() -> StrategyRegistry:
    """Build a registry with built-in strategy factories."""
    registry = StrategyRegistry()

    registry.register("spatial", "ip1", lambda _: IP1SpatialStrategy())
    registry.register("spatial", "ip2", lambda _: IP2SpatialStrategy())
    registry.register("spatial", "iss1", lambda _: ISS1SpatialStrategy())

    registry.register("source", "gauss", lambda _: GaussSourceStrategy())
    registry.register("source", "laplace", lambda _: LaplaceSourceStrategy())
    registry.register("source", "ilrma_nmf", lambda _: ILRMANMFSourceStrategy())

    registry.register("covariance", "batch", lambda _: BatchCovarianceStrategy())
    registry.register(
        "covariance",
        "ema",
        lambda params: EMACovarianceStrategy(alpha=params.get("alpha")),
    )

    registry.register("nmf", "multiplicative", lambda _: MultiplicativeNMFStrategy())
    registry.register("assignment", "modulo", lambda _: ModuloAssignmentStrategy())

    return registry


__all__ = [
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
    "StrategyRegistry",
    "StrategyRegistryError",
    "default_strategy_registry",
]
