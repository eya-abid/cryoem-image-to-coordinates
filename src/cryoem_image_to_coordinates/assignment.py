"""Target-space PCA diagnostics for paired particle-specific assignment."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .metrics import as_coordinate_array


def assignment_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    components: int = 10,
    chunk_size: int = 256,
    seed: int = 42,
) -> dict[str, float | int]:
    """Rank each paired target among all targets by target-PCA z-distance.

    PCA is fitted only on the supplied target population. This is a diagnostic of
    assignment within that target library, not an independent structural truth test.
    """
    prediction_array = as_coordinate_array(predictions)
    target_array = as_coordinate_array(targets)
    if prediction_array.shape != target_array.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction_array.shape} != {target_array.shape}")
    target_flat = target_array.reshape(target_array.shape[0], -1)
    prediction_flat = prediction_array.reshape(prediction_array.shape[0], -1)
    if not 1 <= components < min(target_flat.shape):
        raise ValueError(f"Invalid component count {components} for target shape {target_flat.shape}")

    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    target_scores = pca.fit_transform(target_flat)
    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
    target_z = target_scores / scale
    prediction_z = pca.transform(prediction_flat) / scale

    paired = np.linalg.norm(prediction_z - target_z, axis=1)
    target_squared = np.sum(target_z * target_z, axis=1)
    ranks = np.empty(target_z.shape[0], dtype=np.int64)
    for start in range(0, target_z.shape[0], chunk_size):
        stop = min(start + chunk_size, target_z.shape[0])
        query = prediction_z[start:stop]
        squared = np.sum(query * query, axis=1, keepdims=True) + target_squared[None, :] - 2.0 * query @ target_z.T
        squared = np.maximum(squared, 0.0)
        row_indices = np.arange(stop - start)
        paired_squared = squared[row_indices, np.arange(start, stop)]
        tolerance = np.maximum(1e-12, np.abs(paired_squared) * 1e-10)
        ranks[start:stop] = 1 + np.sum(squared < (paired_squared - tolerance)[:, None], axis=1)

    return {
        "n": int(target_z.shape[0]),
        "pca_components": int(components),
        "paired_distance_mean": float(paired.mean()),
        "paired_distance_median": float(np.median(paired)),
        "paired_target_rank_mean": float(ranks.mean()),
        "paired_target_rank_median": float(np.median(ranks)),
        "top1_rate": float(np.mean(ranks <= 1)),
        "top10_rate": float(np.mean(ranks <= 10)),
        "top100_rate": float(np.mean(ranks <= 100)),
        "target_explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
    }
