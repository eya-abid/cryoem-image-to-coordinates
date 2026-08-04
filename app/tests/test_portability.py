import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
inference = importlib.import_module("inference")
SYSTEMS = inference.SYSTEMS
PredictionService = inference.PredictionService
normalize_minmax = inference.normalize_minmax
normalize_zscore = inference.normalize_zscore


def test_default_checkpoint_paths_are_repository_relative():
    for config in SYSTEMS.values():
        for checkpoint in (
            config.checkpoint,
            config.staged_encoder_checkpoint,
            config.staged_decoder_checkpoint,
        ):
            assert APP_DIR in checkpoint.parents


def test_catalog_starts_without_private_checkpoints_or_examples():
    catalog = PredictionService().catalog()
    assert [entry["key"] for entry in catalog] == ["ak", "her2_synthetic", "her2_experimental"]
    assert all(set(entry["availability"]) == {"direct", "staged", "examples"} for entry in catalog)


def test_system_specific_normalization():
    image = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    minmax, minmax_stats = normalize_minmax(image)
    zscore, zscore_stats = normalize_zscore(image)
    assert np.isclose(minmax.min(), 0.0)
    assert np.isclose(minmax.max(), 1.0)
    assert np.isclose(zscore.mean(), 0.0, atol=1e-6)
    assert np.isclose(zscore.std(), 1.0, atol=1e-6)
    assert minmax_stats["input_max"] > minmax_stats["input_min"]
    assert zscore_stats["clip_high"] > zscore_stats["clip_low"]


def test_public_assets_match_declared_coordinate_counts():
    for key, config in SYSTEMS.items():
        mean = np.load(APP_DIR / "assets" / key / "training_target_mean.npy")
        assert mean.shape == (config.atom_count, 3)


def test_missing_checkpoint_error_does_not_expose_private_paths():
    service = PredictionService().system("ak")
    with pytest.raises(FileNotFoundError, match=r"app/CHECKPOINTS\.md") as error:
        service.load_model()
    assert "/run/media" not in str(error.value)
