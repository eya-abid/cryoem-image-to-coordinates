import numpy as np
import pytest

from cryoem_image_to_coordinates.metrics import aligned_rmsd, as_coordinate_array, raw_rmsd, summarize


def test_raw_rmsd_known_translation() -> None:
    target = np.zeros((2, 4, 3), dtype=np.float64)
    prediction = target + np.array([1.0, 2.0, 2.0])
    assert np.allclose(raw_rmsd(prediction, target), 3.0)


def test_aligned_rmsd_removes_rigid_transform() -> None:
    target = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    prediction = target @ rotation.T + np.array([5.0, -2.0, 7.0])
    assert raw_rmsd(prediction, target)[0] > 1.0
    assert aligned_rmsd(prediction, target)[0] < 1e-10


def test_flattened_coordinates_are_supported() -> None:
    array = np.zeros((5, 12), dtype=np.float32)
    assert as_coordinate_array(array).shape == (5, 4, 3)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        raw_rmsd(np.zeros((2, 4, 3)), np.zeros((3, 4, 3)))


def test_summary() -> None:
    summary = summarize(np.array([1.0, 2.0, 3.0]))
    assert summary["n"] == 3
    assert summary["mean"] == 2.0
