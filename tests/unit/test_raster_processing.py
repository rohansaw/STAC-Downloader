import numpy as np
import pytest

from stac_downloader.raster_processing import apply_mask, is_binary


# Tests for is_binary
def test_is_binary_all_zeros():
    raster = np.zeros((5, 5), dtype=int)
    assert is_binary(raster) is True


def test_is_binary_all_ones():
    raster = np.ones((5, 5), dtype=int)
    assert is_binary(raster) is True


def test_is_binary_mix_zeros_ones():
    raster = np.array([[0, 1], [1, 0]])
    assert is_binary(raster) is True


def test_is_binary_with_other_values():
    raster = np.array([[0, 1, 2]])
    assert is_binary(raster) is False


def test_is_binary_with_negative_value():
    raster = np.array([[0, 1, -1]])
    assert is_binary(raster) is False


# Tests for apply_mask
def test_apply_mask_basic():
    raster = np.array([[1, 2], [3, 4]])
    mask = np.array([[True, False], [False, True]])
    nodata_value = -9999
    expected = np.array([[1, -9999], [-9999, 4]])
    np.testing.assert_array_equal(apply_mask(raster, mask, nodata_value), expected)


def test_apply_mask_all_false():
    raster = np.array([[5, 6], [7, 8]])
    mask = np.array([[False, False], [False, False]])
    nodata_value = -1
    expected = np.full_like(raster, -1)
    np.testing.assert_array_equal(apply_mask(raster, mask, nodata_value), expected)


def test_apply_mask_all_true():
    raster = np.array([[9, 10], [11, 12]])
    mask = np.array([[True, True], [True, True]])
    nodata_value = 0
    expected = raster.copy()
    np.testing.assert_array_equal(apply_mask(raster, mask, nodata_value), expected)


def test_apply_mask_shape_mismatch():
    raster = np.array([[1, 2], [3, 4]])
    mask = np.array([[True, False]])
    with pytest.raises(ValueError):
        apply_mask(raster, mask, nodata_value=0)
