import numpy as np

from oobss import OnlineFrameRequest, OnlineISNMF


def test_online_isnmf_separate_frame_returns_ratio_mask() -> None:
    model = OnlineISNMF(n_components=4, n_features=8, random_state=0)
    frame = np.ones(8, dtype=np.complex128)

    separated, mask = model.separate_frame(frame, n_sources=2)

    assert separated.shape == (2, 8)
    assert mask.shape == (2, 8)
    np.testing.assert_allclose(mask.sum(axis=0), np.ones(8), rtol=1e-6, atol=1e-9)


def test_online_isnmf_process_frame_supports_frame_request() -> None:
    model = OnlineISNMF(n_components=4, n_features=8, random_state=0)
    frame = np.ones(8, dtype=np.complex128)

    mask = model.process_frame(
        frame,
        request=OnlineFrameRequest(n_sources=2, return_mask=True),
    )
    assert mask.shape == (2, 8)
    np.testing.assert_allclose(mask.sum(axis=0), np.ones(8), rtol=1e-6, atol=1e-9)
