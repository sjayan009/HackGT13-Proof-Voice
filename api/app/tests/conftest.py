"""Test fixtures: a tiny random-init SSL detector saved in the real on-disk format, so API/pipeline tests exercise
the exact load path without needing the multi-hundred-MB selected checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

API = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(API))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    import torch
    from transformers import Wav2Vec2Config

    from app.detectors.ssl_model import SSLDetector, save_detector

    torch.manual_seed(0)
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(32, 32, 32, 32, 32, 32, 32), feat_extract_norm="layer",
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4, do_stable_layer_norm=True)
    m = SSLDetector("tiny-test", pretrained=False, config=cfg)
    d = tmp_path_factory.mktemp("tiny_model")
    save_detector(m, d, {"name": "tiny-test", "version": "test"})
    (d / "calibration.json").write_text(json.dumps({"a": 1.0, "b": 0.0, "decision_threshold": 0.5}))
    return d


@pytest.fixture(scope="session")
def detector(tiny_model_dir):
    from app.detectors.primary import PrimaryDetector

    return PrimaryDetector(tiny_model_dir, "cpu")


@pytest.fixture
def speechlike():
    """3.5 s harmonic signal with syllable-rate amplitude modulation (not real speech; deterministic)."""
    sr = 16000
    t = np.arange(int(3.5 * sr)) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    x += 0.01 * np.random.default_rng(0).standard_normal(len(t))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)
