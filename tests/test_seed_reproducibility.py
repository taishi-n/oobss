import numpy as np

from oobss import ILRMA, OnlineILRMA, OnlineISNMF, StreamRequest


def _random_tf(seed: int, n_frame: int, n_freq: int, n_mic: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frame, n_freq, n_mic)) + 1j * rng.standard_normal(
        (n_frame, n_freq, n_mic)
    )


def test_ilrma_reproducible_with_same_random_state() -> None:
    obs = _random_tf(seed=0, n_frame=8, n_freq=10, n_mic=2)

    model_a = ILRMA(obs, n_basis=3, random_state=123)
    out_a = model_a.run(2)

    model_b = ILRMA(obs, n_basis=3, random_state=123)
    out_b = model_b.run(2)

    np.testing.assert_allclose(out_a, out_b, rtol=0.0, atol=0.0)

    out_call_a = model_a(obs, n_iter=2)
    out_call_b = model_b(obs, n_iter=2)
    assert out_call_a.estimate_tf is not None
    assert out_call_b.estimate_tf is not None
    np.testing.assert_allclose(out_call_a.estimate_tf, out_call_b.estimate_tf)


def test_online_ilrma_reproducible_with_same_random_state() -> None:
    rng = np.random.default_rng(1)
    spec = rng.standard_normal((10, 2, 6)) + 1j * rng.standard_normal((10, 2, 6))

    model_a = OnlineILRMA(
        n_mic=2, n_freq=10, n_bases=4, inner_iter=2, beta=2, random_state=7
    )
    model_b = OnlineILRMA(
        n_mic=2, n_freq=10, n_bases=4, inner_iter=2, beta=2, random_state=7
    )

    out_a = model_a(spec, request=StreamRequest(frame_axis=2))
    out_b = model_b(spec, request=StreamRequest(frame_axis=2))
    assert out_a.estimate_tf is not None
    assert out_b.estimate_tf is not None
    np.testing.assert_allclose(out_a.estimate_tf, out_b.estimate_tf, rtol=0.0, atol=0.0)


def test_online_isnmf_reproducible_with_same_random_state() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 7)) + 1j * rng.standard_normal((10, 7))

    model_a = OnlineISNMF(
        n_components=6,
        n_features=10,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=9,
    )
    model_b = OnlineISNMF(
        n_components=6,
        n_features=10,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=9,
    )

    out_a = model_a.process_stream_tf(x, request=StreamRequest(frame_axis=1))
    out_b = model_b.process_stream_tf(x, request=StreamRequest(frame_axis=1))
    assert out_a.estimate_tf is not None
    assert out_b.estimate_tf is not None
    np.testing.assert_allclose(out_a.estimate_tf, out_b.estimate_tf, rtol=0.0, atol=0.0)


def test_online_isnmf_differs_with_different_random_state() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 6)) + 1j * rng.standard_normal((8, 6))

    model_a = OnlineISNMF(
        n_components=4,
        n_features=8,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=11,
    )
    model_b = OnlineISNMF(
        n_components=4,
        n_features=8,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=12,
    )

    out_a = model_a.process_stream_tf(x, request=StreamRequest(frame_axis=1))
    out_b = model_b.process_stream_tf(x, request=StreamRequest(frame_axis=1))
    assert out_a.estimate_tf is not None
    assert out_b.estimate_tf is not None
    assert not np.allclose(out_a.estimate_tf, out_b.estimate_tf)


def test_online_ilrma_reset_restores_initial_state_with_seed() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((10, 2)) + 1j * rng.standard_normal((10, 2))

    model = OnlineILRMA(
        n_mic=2,
        n_freq=10,
        n_bases=3,
        inner_iter=2,
        beta=2,
        random_state=21,
    )
    basis_initial = model.basis.copy()
    demix_initial = model.demix.copy()

    model.process_frame(x)
    model.reset()

    np.testing.assert_allclose(model.basis, basis_initial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model.demix, demix_initial, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(model._batch_counter, np.zeros((2,), dtype=np.int64))
    np.testing.assert_array_equal(model._t, np.zeros((2,), dtype=np.int64))


def test_online_isnmf_reset_restores_initial_state_with_seed() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(10) + 1j * rng.standard_normal(10)

    model = OnlineISNMF(
        n_components=6,
        n_features=10,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=31,
    )
    W_initial = model.W.copy()
    A_initial = model.A.copy()
    B_initial = model.B.copy()

    model.process_frame(x)
    model.reset()

    np.testing.assert_allclose(model.W, W_initial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model.A, A_initial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model.B, B_initial, rtol=0.0, atol=0.0)
    assert model._batch_counter == 0
    assert model._t == 0
