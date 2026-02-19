import numpy as np

from oobss.experiments.postprocess import (
    ParameterEstimationResult,
    gaussian_source_model_weight,
    mixing_matrix_from_demixing_for_reference,
    separate_with_reference,
)


class _DummyIstft:
    def istft(self, spectrogram: np.ndarray) -> np.ndarray:
        return np.real(np.sum(spectrogram, axis=1))


def _batch_demixing(n_freq: int = 3) -> np.ndarray:
    mixing = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [0.2 + 0.0j, 1.2 + 0.0j],
        ]
    )
    demix = np.linalg.inv(mixing)
    return np.broadcast_to(demix[None, :, :], (n_freq, 2, 2)).copy()


def test_mixing_matrix_from_demixing_normalizes_reference_row_batch() -> None:
    demixing = _batch_demixing(n_freq=4)
    mixing = mixing_matrix_from_demixing_for_reference(demixing, ref_mic=0)
    assert mixing.shape == (4, 2, 2)
    np.testing.assert_allclose(mixing[:, 0, :], 1.0, rtol=0.0, atol=1.0e-12)


def test_mixing_matrix_from_demixing_normalizes_reference_row_online() -> None:
    demixing = np.stack([_batch_demixing(n_freq=3), _batch_demixing(n_freq=3)], axis=0)
    mixing = mixing_matrix_from_demixing_for_reference(demixing, ref_mic=0)
    assert mixing.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(mixing[:, :, 0, :], 1.0, rtol=0.0, atol=1.0e-12)


def test_separate_with_reference_returns_expected_shapes() -> None:
    rng = np.random.default_rng(0)
    demixed_tf_raw = rng.standard_normal((6, 3, 2)) + 1j * rng.standard_normal(
        (6, 3, 2)
    )
    params = ParameterEstimationResult(
        method_id="batch_auxiva",
        demixed_tf_raw=demixed_tf_raw,
        demixing_matrix=_batch_demixing(n_freq=3),
        source_model=np.ones_like(demixed_tf_raw, dtype=np.float64),
        stft=_DummyIstft(),
    )

    out = separate_with_reference(params, ref_mic=0, n_samples=5)
    assert out.ref_mic == 0
    assert out.projected_tf.shape == (6, 3, 2)
    assert out.mixing_matrix.shape == (3, 2, 2)
    assert out.estimate.shape == (2, 5)

    expected = np.real(np.sum(out.projected_tf.transpose(2, 1, 0), axis=1))[:, :5]
    np.testing.assert_allclose(out.estimate, expected, rtol=0.0, atol=1.0e-12)


def test_gaussian_source_model_weight_broadcasts_frequency_axis() -> None:
    rng = np.random.default_rng(1)
    demixed = rng.standard_normal((4, 5, 2)) + 1j * rng.standard_normal((4, 5, 2))
    source_model = gaussian_source_model_weight(demixed)
    assert source_model.shape == demixed.shape
    np.testing.assert_allclose(source_model[:, 0, :], source_model[:, 1, :])
