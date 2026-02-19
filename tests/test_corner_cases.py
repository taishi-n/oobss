import numpy as np
import pytest

from oobss import AuxIVA, ILRMA, OnlineAuxIVA, OnlineILRMA, OnlineISNMF, StreamRequest


def test_streaming_separator_rejects_empty_stream() -> None:
    model = OnlineAuxIVA(n_mic=2, n_freq=8, inner_iter=1)
    empty = np.empty((8, 2, 0), dtype=np.complex128)
    with pytest.raises(ValueError, match="at least one frame"):
        model.process_stream_tf(empty, request=StreamRequest(frame_axis=2))


def test_batch_separators_stable_on_single_source_tiny_input() -> None:
    rng = np.random.default_rng(0)
    obs = (rng.standard_normal((6, 8, 1)) + 1j * rng.standard_normal((6, 8, 1))) * 1e-6

    auxiva = AuxIVA(obs)
    aux_out = auxiva.run(1)
    assert aux_out.shape == obs.shape
    assert np.all(np.isfinite(aux_out))

    ilrma = ILRMA(obs, n_basis=2, random_state=0)
    ilrma_out = ilrma.run(1)
    assert ilrma_out.shape == obs.shape
    assert np.all(np.isfinite(ilrma_out))


def test_online_separators_handle_single_frame_tiny_input() -> None:
    rng = np.random.default_rng(1)
    spec = (rng.standard_normal((8, 2, 1)) + 1j * rng.standard_normal((8, 2, 1))) * 1e-8

    online_auxiva = OnlineAuxIVA(n_mic=2, n_freq=8, inner_iter=1)
    out_auxiva = online_auxiva.process_stream_tf(
        spec,
        request=StreamRequest(frame_axis=2),
    )
    assert out_auxiva.estimate_tf is not None
    out_auxiva_tf = out_auxiva.estimate_tf
    assert out_auxiva_tf.shape == (8, 2, 1)
    assert np.all(np.isfinite(out_auxiva_tf))

    online_ilrma = OnlineILRMA(
        n_mic=2,
        n_freq=8,
        n_bases=3,
        beta=2,
        inner_iter=1,
        random_state=0,
    )
    out_ilrma = online_ilrma.process_stream_tf(
        spec,
        request=StreamRequest(frame_axis=2),
    )
    assert out_ilrma.estimate_tf is not None
    out_ilrma_tf = out_ilrma.estimate_tf
    assert out_ilrma_tf.shape == (8, 2, 1)
    assert np.all(np.isfinite(out_ilrma_tf))

    x = (rng.standard_normal((8, 1)) + 1j * rng.standard_normal((8, 1))) * 1e-8
    online_isnmf = OnlineISNMF(
        n_components=4,
        n_features=8,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=0,
    )
    out_isnmf = online_isnmf.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=1),
    )
    assert out_isnmf.estimate_tf is not None
    out_isnmf_tf = out_isnmf.estimate_tf
    assert out_isnmf_tf.shape == (2, 8, 1)
    assert np.all(np.isfinite(out_isnmf_tf))


def test_online_isnmf_component_mapping_validation() -> None:
    model = OnlineISNMF(n_components=4, n_features=8, random_state=0)
    frame = np.ones(8, dtype=np.complex128)

    with pytest.raises(ValueError, match="length K"):
        model.separate_frame(
            frame,
            n_sources=2,
            component_to_source=np.array([0, 1, 0], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="out-of-range"):
        model.separate_frame(
            frame,
            n_sources=2,
            component_to_source=np.array([0, 1, 2, 0], dtype=np.int64),
        )


def test_batch_run_zero_iterations_returns_initial_estimate() -> None:
    rng = np.random.default_rng(2)
    obs = rng.standard_normal((5, 6, 2)) + 1j * rng.standard_normal((5, 6, 2))

    auxiva = AuxIVA(obs)
    initial = auxiva.get_estimate().copy()
    out = auxiva.run(0)
    np.testing.assert_allclose(out, initial)


def test_online_isnmf_zero_input_returns_zero_separation() -> None:
    model = OnlineISNMF(
        n_components=4,
        n_features=8,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=0,
    )
    x = np.zeros((8, 3), dtype=np.complex128)
    out = model.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=1),
    )
    assert out.estimate_tf is not None
    out_tf = out.estimate_tf
    assert out_tf.shape == (2, 8, 3)
    np.testing.assert_allclose(out_tf, 0.0, atol=1e-12, rtol=0.0)


def test_online_auxiva_online_ilrma_zero_input_returns_finite() -> None:
    x = np.zeros((8, 2, 2), dtype=np.complex128)

    online_auxiva = OnlineAuxIVA(n_mic=2, n_freq=8, inner_iter=1)
    out_aux = online_auxiva.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=2),
    )
    assert out_aux.estimate_tf is not None
    out_aux_tf = out_aux.estimate_tf
    assert out_aux_tf.shape == (8, 2, 2)
    assert np.all(np.isfinite(out_aux_tf))

    online_ilrma = OnlineILRMA(
        n_mic=2,
        n_freq=8,
        n_bases=3,
        beta=2,
        inner_iter=1,
        random_state=0,
    )
    out_ilr = online_ilrma.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=2),
    )
    assert out_ilr.estimate_tf is not None
    out_ilr_tf = out_ilr.estimate_tf
    assert out_ilr_tf.shape == (8, 2, 2)
    assert np.all(np.isfinite(out_ilr_tf))


def test_online_isnmf_supports_single_frequency_bin() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 5)) + 1j * rng.standard_normal((1, 5))
    model = OnlineISNMF(
        n_components=2,
        n_features=1,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=0,
    )
    out = model.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=1),
    )
    assert out.estimate_tf is not None
    out_tf = out.estimate_tf
    assert out_tf.shape == (2, 1, 5)
    assert np.all(np.isfinite(out_tf))


def test_online_auxiva_reset_restores_identity_demix() -> None:
    rng = np.random.default_rng(5)
    frame = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))

    model = OnlineAuxIVA(n_mic=2, n_freq=8, inner_iter=1)
    demix_initial = model.demix.copy()

    model.process_frame(frame)
    model.reset()

    np.testing.assert_allclose(model.demix, demix_initial, rtol=0.0, atol=0.0)
