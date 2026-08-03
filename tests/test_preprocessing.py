import numpy as np
import pytest

from cryoem_image_to_coordinates.preprocessing import describe_images, minmax_image, prepare_image, zscore_image


def test_minmax_image_bounds() -> None:
    result = minmax_image(np.array([[1.0, 2.0], [3.0, 5.0]]))
    assert result.dtype == np.float32
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)


def test_constant_normalization_is_finite_zero() -> None:
    image = np.ones((4, 4), dtype=np.float32)
    assert np.all(minmax_image(image) == 0)
    assert np.all(zscore_image(image) == 0)


def test_zscore_image_statistics() -> None:
    result = zscore_image(np.arange(16, dtype=np.float32).reshape(4, 4))
    assert abs(float(result.mean())) < 1e-6
    assert np.isclose(result.std(), 1.0, atol=1e-6)


def test_normalization_must_be_explicit() -> None:
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    assert np.array_equal(prepare_image(image, "none"), image)
    with pytest.raises(ValueError):
        prepare_image(image, "automatic")  # type: ignore[arg-type]


def test_describe_images() -> None:
    summary = describe_images(np.zeros((3, 8, 8), dtype=np.float32))
    assert summary["shape"] == [3, 8, 8]
    assert summary["sampled_images"] == 3
