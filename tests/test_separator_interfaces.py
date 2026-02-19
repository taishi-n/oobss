import numpy as np
import pytest

from oobss import (
    AuxIVA,
    BatchRequest,
    ILRMA,
    OnlineAuxIVA,
    OnlineILRMA,
    OnlineISNMF,
    StreamRequest,
    StreamingSeparatorState,
)
from oobss.separators.strategies import (
    BatchCovarianceStrategy,
    GaussSourceStrategy,
    IP1SpatialStrategy,
    ModuloAssignmentStrategy,
    MultiplicativeNMFStrategy,
)


def _random_tf(
    rng: np.random.Generator, n_frame: int, n_freq: int, n_mic: int
) -> np.ndarray:
    return rng.standard_normal((n_frame, n_freq, n_mic)) + 1j * rng.standard_normal(
        (n_frame, n_freq, n_mic)
    )


def test_batch_separators_support_run_interface() -> None:
    rng = np.random.default_rng(0)
    obs = _random_tf(rng, n_frame=6, n_freq=8, n_mic=2)

    auxiva = AuxIVA(obs)
    aux_out = auxiva.run(1)
    assert aux_out.shape == obs.shape

    ilrma = ILRMA(obs, n_basis=3)
    ilrma_out = ilrma.run(1)
    assert ilrma_out.shape == obs.shape


def test_streaming_separators_support_process_stream_interface() -> None:
    rng = np.random.default_rng(0)
    n_freq, n_mic, n_frame = 8, 2, 5
    spec = rng.standard_normal((n_freq, n_mic, n_frame)) + 1j * rng.standard_normal(
        (n_freq, n_mic, n_frame)
    )

    auxiva = OnlineAuxIVA(n_mic=n_mic, n_freq=n_freq, inner_iter=1)
    aux_out = auxiva.process_stream_tf(
        spec,
        request=StreamRequest(frame_axis=2),
    )
    assert aux_out.estimate_tf is not None
    aux_est = aux_out.estimate_tf
    assert aux_est.shape == spec.shape

    ilrma = OnlineILRMA(n_mic=n_mic, n_freq=n_freq, n_bases=3, beta=2, inner_iter=1)
    ilrma_out = ilrma.process_stream_tf(
        spec,
        request=StreamRequest(frame_axis=2),
    )
    assert ilrma_out.estimate_tf is not None
    ilrma_est = ilrma_out.estimate_tf
    assert ilrma_est.shape == spec.shape


def test_online_isnmf_supports_process_stream_interface() -> None:
    rng = np.random.default_rng(0)
    n_freq, n_frame, n_sources = 8, 6, 2
    x = rng.standard_normal((n_freq, n_frame)) + 1j * rng.standard_normal(
        (n_freq, n_frame)
    )

    isnmf = OnlineISNMF(
        n_components=4,
        n_features=n_freq,
        n_sources=n_sources,
        inner_iter=2,
        beta=2,
        random_state=0,
    )
    out = isnmf.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=1),
    )
    assert out.estimate_tf is not None
    estimate_tf = out.estimate_tf
    assert estimate_tf.shape == (n_sources, n_freq, n_frame)


def test_streaming_separator_supports_stream_request() -> None:
    rng = np.random.default_rng(2)
    n_freq, n_frame = 8, 5
    x = rng.standard_normal((n_freq, n_frame)) + 1j * rng.standard_normal(
        (n_freq, n_frame)
    )
    isnmf = OnlineISNMF(
        n_components=4,
        n_features=n_freq,
        n_sources=2,
        inner_iter=2,
        beta=2,
        random_state=0,
    )
    out = isnmf.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=1, n_sources=2, return_mask=True),
    )
    assert out.estimate_tf is not None
    assert out.estimate_tf.shape == (2, n_freq, n_frame)


def test_auxiva_fit_transform_tf_binds_mixture() -> None:
    rng = np.random.default_rng(10)
    obs0 = _random_tf(rng, n_frame=4, n_freq=6, n_mic=2)
    obs1 = _random_tf(rng, n_frame=4, n_freq=6, n_mic=2)

    separator = AuxIVA(obs0)
    output = separator.fit_transform_tf(obs1, request=BatchRequest())

    assert output.estimate_tf is not None
    np.testing.assert_allclose(output.estimate_tf, obs1)
    np.testing.assert_allclose(separator.observations, obs1)


def test_batch_separator_supports_forward_call_style() -> None:
    rng = np.random.default_rng(12)
    obs = _random_tf(rng, n_frame=4, n_freq=6, n_mic=2)
    separator = AuxIVA(obs)

    out = separator.forward(obs, n_iter=1)
    assert out.estimate_tf is not None
    assert out.estimate_tf.shape == obs.shape


def test_auxiva_supports_explicit_strategy_injection() -> None:
    rng = np.random.default_rng(14)
    obs = _random_tf(rng, n_frame=5, n_freq=8, n_mic=2)
    separator = AuxIVA(
        obs,
        spatial=IP1SpatialStrategy(),
        source=GaussSourceStrategy(),
        covariance=BatchCovarianceStrategy(),
    )

    out = separator.forward(obs, n_iter=1)
    assert out.estimate_tf is not None
    assert out.estimate_tf.shape == obs.shape


def test_online_isnmf_supports_explicit_strategy_injection() -> None:
    rng = np.random.default_rng(15)
    n_freq, n_frame = 8, 4
    x = rng.standard_normal((n_freq, n_frame)) + 1j * rng.standard_normal(
        (n_freq, n_frame)
    )

    isnmf = OnlineISNMF(
        n_components=4,
        n_features=n_freq,
        n_sources=2,
        nmf=MultiplicativeNMFStrategy(),
        assignment=ModuloAssignmentStrategy(),
        random_state=0,
    )
    out = isnmf.process_stream_tf(
        x,
        request=StreamRequest(frame_axis=1),
    )
    assert out.estimate_tf is not None
    assert out.estimate_tf.shape == (2, n_freq, n_frame)


def test_online_auxiva_forward_streaming_supports_state_roundtrip() -> None:
    rng = np.random.default_rng(13)
    n_freq, n_mic = 8, 2
    frame0 = rng.standard_normal((n_freq, n_mic)) + 1j * rng.standard_normal(
        (n_freq, n_mic)
    )
    frame1 = rng.standard_normal((n_freq, n_mic)) + 1j * rng.standard_normal(
        (n_freq, n_mic)
    )

    model = OnlineAuxIVA(n_mic=n_mic, n_freq=n_freq, inner_iter=1)
    y0, state0 = model.forward_streaming(frame0)
    assert y0.shape == (n_freq, n_mic)
    assert isinstance(state0, StreamingSeparatorState)
    assert state0.demix_filter is not None

    model2 = OnlineAuxIVA(n_mic=n_mic, n_freq=n_freq, inner_iter=1)
    y1_a, _ = model.forward_streaming(frame1)
    y1_b, _ = model2.forward_streaming(frame1, state=state0)
    np.testing.assert_allclose(y1_a, y1_b, atol=1e-8, rtol=1e-6)


def test_auxiva_rejects_legacy_update_keyword_arguments() -> None:
    rng = np.random.default_rng(16)
    obs = _random_tf(rng, n_frame=3, n_freq=4, n_mic=2)

    with pytest.raises(TypeError):
        AuxIVA(obs, update_demix_filter="IP1")  # type: ignore[call-arg]


def test_iterative_separators_reject_time_input_by_default() -> None:
    rng = np.random.default_rng(11)
    obs = _random_tf(rng, n_frame=4, n_freq=6, n_mic=2)
    separator = ILRMA(obs, n_basis=2)
    mixture_time = np.zeros((64, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="does not support time-domain input"):
        separator.fit_transform_time(
            mixture_time,
            request=BatchRequest(sample_rate=8000),
        )
