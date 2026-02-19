import numpy as np

from oobss.separators.strategies import (
    BssEvalPermutationStrategy,
    IdentityNormalization,
    IdentityPermutationStrategy,
    ProjectionBackDemixReconstructionStrategy,
    ProjectionBackNormalization,
    ScoreMatrixPermutationStrategy,
)
from oobss.separators.core import (
    NormalizationRequest,
    PermutationRequest,
    ReconstructionRequest,
)
from oobss.separators.utils import projection_back


def test_identity_normalization_returns_input() -> None:
    rng = np.random.default_rng(0)
    estimate = rng.standard_normal((4, 8, 2)) + 1j * rng.standard_normal((4, 8, 2))
    strategy = IdentityNormalization()
    out = strategy.apply(NormalizationRequest(estimate=estimate))
    np.testing.assert_allclose(out, estimate)


def test_projection_back_normalization_matches_utility() -> None:
    rng = np.random.default_rng(0)
    estimate = rng.standard_normal((5, 8, 2)) + 1j * rng.standard_normal((5, 8, 2))
    observations = rng.standard_normal((5, 8, 2)) + 1j * rng.standard_normal((5, 8, 2))
    demix = np.tile(np.eye(2, dtype=complex), (8, 1, 1))

    strategy = ProjectionBackNormalization(ref_mic=0)
    out = strategy.apply(
        NormalizationRequest(
            estimate=estimate,
            observations=observations,
            demix_filter=demix,
        )
    )
    ref = projection_back(estimate, demix, ref_mic=0)
    np.testing.assert_allclose(out, ref)


def test_projection_back_normalization_without_observations() -> None:
    rng = np.random.default_rng(11)
    estimate = rng.standard_normal((5, 8, 2)) + 1j * rng.standard_normal((5, 8, 2))
    demix = np.tile(np.eye(2, dtype=complex), (8, 1, 1))

    strategy = ProjectionBackNormalization(ref_mic=0)
    out = strategy.apply(
        NormalizationRequest(
            estimate=estimate,
            demix_filter=demix,
        )
    )
    ref = projection_back(estimate, demix, ref_mic=0)
    np.testing.assert_allclose(out, ref)


def test_projection_back_reconstruction_uses_shared_utility() -> None:
    rng = np.random.default_rng(7)
    mixture = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))
    demix = np.tile(np.eye(2, dtype=complex), (8, 1, 1))

    strategy = ProjectionBackDemixReconstructionStrategy(ref_mic=0)
    out = strategy.reconstruct(
        ReconstructionRequest(
            mixture=mixture,
            demix_filter=demix,
        )
    ).estimate
    demixed = (demix @ mixture[:, :, None])[:, :, 0]
    ref = projection_back(demixed, demix, ref_mic=0)
    np.testing.assert_allclose(out, ref)


def test_identity_permutation_strategy() -> None:
    score = np.array([[0.1, 0.9], [0.8, 0.2]])
    strategy = IdentityPermutationStrategy()
    perm = strategy.solve(PermutationRequest(score=score))
    np.testing.assert_array_equal(perm, np.array([0, 1], dtype=np.int64))


def test_score_matrix_permutation_strategy() -> None:
    score = np.array([[0.1, 0.9], [0.8, 0.2]])
    strategy = ScoreMatrixPermutationStrategy()
    perm = strategy.solve(PermutationRequest(score=score))
    np.testing.assert_array_equal(perm, np.array([1, 0], dtype=np.int64))


def test_bss_eval_permutation_strategy_prefers_default() -> None:
    strategy = BssEvalPermutationStrategy(filter_length=1)
    default_perm = np.array([1, 0], dtype=np.int64)
    perm = strategy.solve(
        PermutationRequest(
            score=np.zeros((2,), dtype=float),
            default_perm=default_perm,
            filter_length=1,
        )
    )
    np.testing.assert_array_equal(perm, default_perm)
