import numpy as np
import pytest

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.transform import from_origin

from stac_downloader import raster_processing as rp


# Tests for is_binary
def test_is_binary_all_zeros():
    raster = np.zeros((5, 5), dtype=int)
    assert rp.is_binary(raster) is True


def test_is_binary_all_ones():
    raster = np.ones((5, 5), dtype=int)
    assert rp.is_binary(raster) is True


def test_is_binary_mix_zeros_ones():
    raster = np.array([[0, 1], [1, 0]])
    assert rp.is_binary(raster) is True


def test_is_binary_with_other_values():
    raster = np.array([[0, 1, 2]])
    assert rp.is_binary(raster) is False


def test_is_binary_with_negative_value():
    raster = np.array([[0, 1, -1]])
    assert rp.is_binary(raster) is False


# Tests for apply_mask
def test_apply_mask_basic():
    raster = np.array([[1, 2], [3, 4]])
    mask = np.array([[True, False], [False, True]])
    nodata_value = -9999
    expected = np.array([[1, -9999], [-9999, 4]])
    np.testing.assert_array_equal(rp.apply_mask(raster, mask, nodata_value), expected)


def test_apply_mask_all_false():
    raster = np.array([[5, 6], [7, 8]])
    mask = np.array([[False, False], [False, False]])
    nodata_value = -1
    expected = np.full_like(raster, -1)
    np.testing.assert_array_equal(rp.apply_mask(raster, mask, nodata_value), expected)


def test_apply_mask_all_true():
    raster = np.array([[9, 10], [11, 12]])
    mask = np.array([[True, True], [True, True]])
    nodata_value = 0
    expected = raster.copy()
    np.testing.assert_array_equal(rp.apply_mask(raster, mask, nodata_value), expected)


def test_apply_mask_shape_mismatch():
    raster = np.array([[1, 2], [3, 4]])
    mask = np.array([[True, False]])
    with pytest.raises(ValueError):
        rp.apply_mask(raster, mask, nodata_value=0)


# ---------- Helpers ----------
def _write_single_band_geotiff(tmp_path, arr, crs="EPSG:4326", transform=None, dtype=None):
    if transform is None:
        transform = from_origin(0, 5, 1, 1)
    if dtype is None:
        dtype = arr.dtype
    path = tmp_path / "src.tif"
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": False,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


# ---------- ResamplingMethod ----------
def test_resamplingmethod_from_string_valid():
    assert rp.ResamplingMethod.from_string("nearest") is rp.ResamplingMethod.NEAREST
    assert rp.ResamplingMethod.from_string("BILINEAR") is rp.ResamplingMethod.BILINEAR


def test_resamplingmethod_from_string_invalid():
    with pytest.raises(ValueError):
        rp.ResamplingMethod.from_string("cubic")


# ---------- is_binary ----------
@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([[0, 0], [0, 0]]), True),
        (np.array([[1, 1], [1, 1]]), True),
        (np.array([[0, 1], [1, 0]]), True),
        (np.array([[0, 2], [1, 0]]), False),
        (np.array([[2, 2], [2, 2]]), False),
    ],
)
def test_is_binary(values, expected):
    assert rp.is_binary(values) is expected


# ---------- apply_mask ----------
def test_apply_mask_happy_path():
    raster = np.array([[10, 20], [30, 40]], dtype=np.int16)
    mask = np.array([[True, False], [True, False]])
    out = rp.apply_mask(raster, mask, nodata_value=-9999)
    expected = np.array([[10, -9999], [30, -9999]], dtype=np.int16)
    np.testing.assert_array_equal(out, expected)


def test_apply_mask_shape_mismatch_raises():
    raster = np.zeros((2, 2), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        rp.apply_mask(raster, mask, nodata_value=0)


# ---------- resample_raster ----------
def test_resample_raster_noop_same_res_and_crs(tmp_path):
    arr = (np.arange(25, dtype=np.uint16).reshape(5, 5))
    path = _write_single_band_geotiff(tmp_path, arr, crs="EPSG:4326", transform=from_origin(0, 5, 1, 1))

    out, profile = rp.resample_raster(
        raster_path=str(path),
        target_resolution=1.0,
        resampling_method=rp.ResamplingMethod.NEAREST,
        target_crs="EPSG:4326",
    )
    # No resampling/reprojection -> identical data/shape/CRS/transform
    np.testing.assert_array_equal(out, arr)
    assert profile["crs"].to_string() == "EPSG:4326"
    assert profile["height"] == arr.shape[0]
    assert profile["width"] == arr.shape[1]


def test_resample_raster_resample_only(tmp_path):
    arr = (np.arange(100, dtype=np.float32).reshape(10, 10))
    path = _write_single_band_geotiff(tmp_path, arr, crs="EPSG:4326", transform=from_origin(0, 10, 1, 1), dtype=np.float32)

    out, profile = rp.resample_raster(
        raster_path=str(path),
        target_resolution=2.0,  # coarser resolution -> fewer pixels
        resampling_method=rp.ResamplingMethod.BILINEAR,
        target_crs="EPSG:4326",
    )
    assert profile["crs"].to_string() == "EPSG:4326"
    assert out.shape != arr.shape  # resampled
    assert out.ndim == 2
    assert profile["height"] == out.shape[0]
    assert profile["width"] == out.shape[1]


def test_resample_raster_reproject_changes_crs(tmp_path):
    arr = (np.arange(100, dtype=np.uint16).reshape(10, 10))
    path = _write_single_band_geotiff(tmp_path, arr, crs="EPSG:4326", transform=from_origin(0, 10, 0.1, 0.1))

    out, profile = rp.resample_raster(
        raster_path=str(path),
        target_resolution=1000,  # meters in EPSG:3857
        resampling_method=rp.ResamplingMethod.NEAREST,
        target_crs="EPSG:3857",
    )
    assert profile["crs"].to_string() == "EPSG:3857"
    assert out.shape != arr.shape  # reprojected grid usually changes shape


def test_resample_raster_raises_on_multiband(tmp_path):
    arr = np.arange(25, dtype=np.uint8).reshape(5, 5)
    path = tmp_path / "multi.tif"
    profile = {
        "driver": "GTiff",
        "height": 5,
        "width": 5,
        "count": 2,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 5, 1, 1),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.write(arr, 2)

    with pytest.raises(ValueError, match="more than one band"):
        rp.resample_raster(
            raster_path=str(path),
            target_resolution=1.0,
            resampling_method=rp.ResamplingMethod.NEAREST,
            target_crs="EPSG:4326",
        )


def test_resample_raster_uses_gcps_when_no_crs(monkeypatch):
    """
    Mocks a dataset with no CRS but with GCPs to ensure the fallback path engages.
    """
    class FakeDS:
        count = 1
        res = (1, 1)
        width = 4
        height = 4
        dtypes = ["uint8"]
        profile = {"height": 4, "width": 4, "count": 1, "dtype": "uint8"}

        crs = None
        transform = None  # should not be used since crs is None

        def read(self, idx):
            return (np.arange(16, dtype=np.uint8).reshape(4, 4))

        @property
        def gcps(self):
            # any truthy GCP list + a CRS object
            return (["gcp1", "gcp2"], rio.crs.CRS.from_epsg(4326))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    # Return an affine transform from the GCPs
    monkeypatch.setattr(rp, "from_gcps", lambda gcps: Affine.translation(0, 4) * Affine.scale(1, -1))
    monkeypatch.setattr(rio, "open", lambda *a, **k: FakeDS())

    out, profile = rp.resample_raster(
        raster_path="dummy.tif",
        target_resolution=1.0,
        resampling_method=rp.ResamplingMethod.NEAREST,
        target_crs="EPSG:4326",
    )
    assert isinstance(out, np.ndarray)
    assert profile["crs"].to_string() == "EPSG:4326"
    assert profile["height"] == out.shape[0]
    assert profile["width"] == out.shape[1]


# ---------- save_band ----------
def test_save_band_writes_and_sets_metadata(tmp_path, monkeypatch):
    # patch persist_file to a simple copy
    def _persist(src, dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    monkeypatch.setattr(rp, "persist_file", _persist)

    # Create a minimal profile and raster
    arr = (np.arange(25, dtype=np.uint16).reshape(5, 5))
    transform = from_origin(0, 5, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    out_path = tmp_path / "out.tif"

    rp.save_band(arr, profile, str(out_path), band_name="mask", dtype="uint16")

    assert out_path.exists()
    with rio.open(out_path) as ds:
        np.testing.assert_array_equal(ds.read(1), arr)
        # band description
        assert ds.descriptions[0] == "mask"
        assert ds.dtypes[0] == "uint16"


# ---------- build_bandstacked_vrt ----------
def test_build_bandstacked_vrt_success(tmp_path, monkeypatch):
    out_vrt = tmp_path / "stack.vrt"
    out_tif = tmp_path / "stack.tif"

    calls = []

    def fake_run(cmd, check, capture_output):
        calls.append(cmd)
        # emulate tools by creating the expected files
        if "gdalbuildvrt" in cmd[0]:
            out_vrt.write_text("<VRTDataset/>")
        elif "gdal_translate" in cmd[0]:
            out_tif.write_bytes(b"II*\x00")  # minimal TIFF header bytes
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    band_paths = {"B1": "/tmp/b1.tif", "B2": "/tmp/b2.tif"}
    band_order = ["B1", "B2"]
    result = rp.build_bandstacked_vrt(
        output_file_path=str(out_vrt),
        band_paths=band_paths,
        band_names=band_order,
        create_gtiff=True,
        blocksize=256,
    )

    assert Path(result).exists()
    assert out_tif.exists()
    # ensure order respected
    assert band_paths["B1"] in calls[0]
    assert band_paths["B2"] in calls[0]


def test_build_bandstacked_vrt_gdalbuildvrt_failure(monkeypatch, tmp_path):
    out_vrt = tmp_path / "stack.vrt"

    def fake_run(cmd, check, capture_output):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="gdalbuildvrt failed"):
        rp.build_bandstacked_vrt(
            output_file_path=str(out_vrt),
            band_paths={"B1": "/tmp/b1.tif"},
            band_names=["B1"],
        )


def test_build_bandstacked_vrt_gdal_translate_failure(monkeypatch, tmp_path):
    out_vrt = tmp_path / "stack.vrt"
    out_vrt.write_text("<VRTDataset/>")  # ensure VRT exists so existence check passes

    def fake_run(cmd, check, capture_output):
        if "gdalbuildvrt" in cmd[0]:
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd, stderr=b"translate error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="gdal_translate failed"):
        rp.build_bandstacked_vrt(
            output_file_path=str(out_vrt),
            band_paths={"B1": "/tmp/b1.tif"},
            band_names=["B1"],
            create_gtiff=True,
        )


def test_build_bandstacked_vrt_missing_outputs(tmp_path, monkeypatch):
    out_vrt = tmp_path / "stack.vrt"

    def fake_run(cmd, check, capture_output):
        # Do not create files -> should trigger FileNotFoundError
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError, match="Output VRT file was not created"):
        rp.build_bandstacked_vrt(
            output_file_path=str(out_vrt),
            band_paths={"B1": "/tmp/b1.tif"},
            band_names=["B1"],
        )

    # If VRT exists but GeoTIFF is missing when requested
    out_vrt.write_text("<VRTDataset/>")
    with pytest.raises(FileNotFoundError, match="Output GeoTIFF file was not created"):
        rp.build_bandstacked_vrt(
            output_file_path=str(out_vrt),
            band_paths={"B1": "/tmp/b1.tif"},
            band_names=["B1"],
            create_gtiff=True,
        )
