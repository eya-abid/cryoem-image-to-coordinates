"""Coordinate metrics with explicit raw-frame and rigid-alignment semantics."""

from __future__ import annotations

import numpy as np


def as_coordinate_array(array: np.ndarray) -> np.ndarray:
    """Return coordinates as `(samples, atoms, 3)` float64."""
    coordinates = np.asarray(array, dtype=np.float64)
    if coordinates.ndim == 2:
        if coordinates.shape[1] % 3:
            raise ValueError(f"Flattened coordinate width is not divisible by 3: {coordinates.shape}")
        coordinates = coordinates.reshape(coordinates.shape[0], -1, 3)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError(f"Expected (N,M,3) or (N,3M), got {coordinates.shape}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Coordinates contain NaN or infinite values")
    return coordinates


def _paired(predictions: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = as_coordinate_array(predictions)
    target_array = as_coordinate_array(targets)
    if prediction_array.shape != target_array.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction_array.shape} != {target_array.shape}")
    return prediction_array, target_array


def raw_rmsd(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute per-sample RMSD in the supplied coordinate frame."""
    prediction_array, target_array = _paired(predictions, targets)
    squared_distance = np.sum((prediction_array - target_array) ** 2, axis=2)
    return np.sqrt(squared_distance.mean(axis=1))


def kabsch_align(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigidly align one ordered prediction to its paired target."""
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape or prediction_array.ndim != 2 or prediction_array.shape[1] != 3:
        raise ValueError("Expected paired (atoms,3) arrays")
    prediction_center = prediction_array.mean(axis=0)
    target_center = target_array.mean(axis=0)
    centered_prediction = prediction_array - prediction_center
    centered_target = target_array - target_center
    covariance = centered_prediction.T @ centered_target
    left, _, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ correction @ right_transpose
    return centered_prediction @ rotation + target_center


def aligned_rmsd(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute per-sample RMSD after optimal proper rigid-body alignment."""
    prediction_array, target_array = _paired(predictions, targets)
    values = np.empty(prediction_array.shape[0], dtype=np.float64)
    for index, (prediction, target) in enumerate(zip(prediction_array, target_array)):
        aligned = kabsch_align(prediction, target)
        values[index] = np.sqrt(np.sum((aligned - target) ** 2, axis=1).mean())
    return values


def summarize(values: np.ndarray) -> dict[str, float | int]:
    """Return stable descriptive statistics for a one-dimensional metric array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Expected a non-empty one-dimensional metric array")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
