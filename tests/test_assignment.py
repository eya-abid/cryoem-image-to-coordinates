import numpy as np

from cryoem_image_to_coordinates.assignment import assignment_metrics


def test_perfect_predictions_have_rank_one() -> None:
    rng = np.random.default_rng(42)
    targets = rng.normal(size=(40, 8, 3))
    summary = assignment_metrics(targets.copy(), targets, components=5, chunk_size=7)
    assert summary["paired_target_rank_median"] == 1.0
    assert summary["top1_rate"] == 1.0
    assert summary["top10_rate"] == 1.0
