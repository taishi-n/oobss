from __future__ import annotations

import pytest

from oobss.experiments.config_schema import parse_common_config


def test_parse_common_config_applies_defaults() -> None:
    cfg = parse_common_config(
        {
            "dataset": {"root": "/tmp/dynamic"},
            "runtime": {"workers": 4},
        }
    )
    assert cfg.dataset.type == "torchrir_dynamic"
    assert cfg.dataset.root == "/tmp/dynamic"
    assert cfg.runtime.workers == 4
    assert cfg.stft.fft_size == 2048


def test_parse_common_config_rejects_unknown_key() -> None:
    with pytest.raises(Exception):
        parse_common_config({"dataset": {"unknown_field": 1}})
