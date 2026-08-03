"""Explicit image normalization used by the public inference/evaluation tools."""

from __future__ import annotations

from typing import Literal

import numpy as np

Normalization = Literal["none", "minmax", "zscore"]


def _as_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"Expected an image with at least two dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Image contains NaN or infinite values")
    return array


def minmax_image(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale one image to [0, 1], returning zeros for a constant image."""
    array = _as_float_image(image)
    minimum = float(array.min())
    span = float(array.max()) - minimum
    if span <= eps:
        return np.zeros_like(array)
    return (array - minimum) / span


def zscore_image(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize one image, returning zeros when its variance is negligible."""
    array = _as_float_image(image)
    mean = float(array.mean())
    standard_deviation = float(array.std())
    if standard_deviation <= eps:
        return np.zeros_like(array)
    return (array - mean) / standard_deviation


def prepare_image(image: np.ndarray, normalization: Normalization = "none") -> np.ndarray:
    """Convert an image to float32 and apply the requested explicit normalization."""
    if normalization == "none":
        return _as_float_image(image).copy()
    if normalization == "minmax":
        return minmax_image(image)
    if normalization == "zscore":
        return zscore_image(image)
    raise ValueError(f"Unsupported normalization: {normalization}")


def describe_images(images: np.ndarray, sample_size: int = 1024) -> dict[str, float | int | list[int]]:
    """Describe a saved image array without modifying it."""
    array = np.asarray(images)
    if array.ndim not in {3, 4}:
        raise ValueError(f"Expected (N,H,W) or (N,C,H,W), got {array.shape}")
    count = min(int(array.shape[0]), sample_size)
    sample = np.asarray(array[:count], dtype=np.float32)
    means = sample.reshape(count, -1).mean(axis=1)
    standard_deviations = sample.reshape(count, -1).std(axis=1)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sampled_images": count,
        "minimum": float(sample.min()),
        "maximum": float(sample.max()),
        "mean_of_image_means": float(means.mean()),
        "mean_of_image_standard_deviations": float(standard_deviations.mean()),
    }
