import numpy as np
import pytest

from oobss.experiments.methods import STFTPlan, default_method_runner_registry


def _run_blockbatch(
    method_type: str,
    mix: np.ndarray,
    params: dict[str, object],
) -> np.ndarray:
    registry = default_method_runner_registry()
    runner = registry.resolve(method_type)
    output = runner.run(
        mix,
        params,
        STFTPlan(fft_size=512, hop_size=256, window="hann"),
        16000,
    )
    if output.estimate_time is None:
        raise ValueError("Runner did not return estimate_time")
    return output.estimate_time


def test_blockbatch_runners_do_not_drop_short_tail_frames() -> None:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((8192, 2))

    methods = [
        (
            "blockbatch_auxiva",
            {"block_size": 20, "overlap": 8, "n_iter": 1, "ref_mic": 0},
        ),
        (
            "blockbatch_ilrma",
            {
                "block_size": 20,
                "overlap": 8,
                "n_iter": 1,
                "n_basis": 2,
                "ref_mic": 0,
                "seed": 0,
            },
        ),
    ]

    for method_type, params in methods:
        estimate = _run_blockbatch(method_type, mix, params)
        assert estimate.shape == (mix.shape[1], mix.shape[0])
        assert np.all(np.isfinite(estimate))
        tail = estimate[:, -512:]
        assert not np.allclose(tail, 0.0, atol=1.0e-12, rtol=0.0)


def test_blockbatch_runners_process_inputs_shorter_than_block_size() -> None:
    rng = np.random.default_rng(1)
    mix = rng.standard_normal((4096, 2))

    methods = [
        (
            "blockbatch_auxiva",
            {"block_size": 64, "overlap": 0, "n_iter": 1, "ref_mic": 0},
        ),
        (
            "blockbatch_ilrma",
            {
                "block_size": 64,
                "overlap": 0,
                "n_iter": 1,
                "n_basis": 2,
                "ref_mic": 0,
                "seed": 0,
            },
        ),
    ]

    for method_type, params in methods:
        estimate = _run_blockbatch(method_type, mix, params)
        assert estimate.shape == (mix.shape[1], mix.shape[0])
        assert np.all(np.isfinite(estimate))
        assert not np.allclose(estimate, 0.0, atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("method_type", "params", "message"),
    [
        (
            "blockbatch_auxiva",
            {"block_size": 0, "overlap": 0, "n_iter": 1, "ref_mic": 0},
            "block_size must be > 0",
        ),
        (
            "blockbatch_auxiva",
            {"block_size": 16, "overlap": 16, "n_iter": 1, "ref_mic": 0},
            "overlap must satisfy 0 <= overlap < block_size",
        ),
        (
            "blockbatch_ilrma",
            {
                "block_size": 16,
                "overlap": -1,
                "n_iter": 1,
                "n_basis": 2,
                "ref_mic": 0,
            },
            "overlap must satisfy 0 <= overlap < block_size",
        ),
        (
            "blockbatch_ilrma",
            {
                "block_size": 16,
                "overlap": 0,
                "n_iter": -1,
                "n_basis": 2,
                "ref_mic": 0,
            },
            "n_iter must be >= 0",
        ),
    ],
)
def test_blockbatch_runner_validates_schedule_params(
    method_type: str,
    params: dict[str, object],
    message: str,
) -> None:
    mix = np.ones((2048, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        _run_blockbatch(method_type, mix, params)
