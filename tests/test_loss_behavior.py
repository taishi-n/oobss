import numpy as np

from oobss import AuxIVA, ILRMA


def _random_observations(seed: int, n_frame: int = 12, n_freq: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frame, n_freq, 2)) + 1j * rng.standard_normal(
        (n_frame, n_freq, 2)
    )


def _assert_monotonic_nonincreasing(losses: list[float], tol: float = 1.0e-8) -> None:
    assert len(losses) >= 2
    deltas = np.diff(np.asarray(losses, dtype=float))
    assert np.all(deltas <= tol), f"loss increased: deltas={deltas.tolist()}"
    assert losses[-1] < losses[0], "final loss must be smaller than initial loss"


def test_auxiva_loss_is_monotonic_nonincreasing() -> None:
    obs = _random_observations(seed=0)
    model = AuxIVA(obs)

    losses = [float(np.real(model.loss))]
    for _ in range(6):
        model.step()
        losses.append(float(np.real(model.loss)))

    assert np.all(np.isfinite(losses))
    _assert_monotonic_nonincreasing(losses)


def test_ilrma_loss_is_monotonic_nonincreasing() -> None:
    obs = _random_observations(seed=1)
    model = ILRMA(
        obs,
        n_basis=3,
        random_state=0,
    )

    losses = [float(np.real(model.loss))]
    for _ in range(6):
        model.step()
        losses.append(float(np.real(model.loss)))

    assert np.all(np.isfinite(losses))
    _assert_monotonic_nonincreasing(losses)
