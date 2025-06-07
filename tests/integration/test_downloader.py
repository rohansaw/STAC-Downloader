import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Fake STAC item and asset structure
class FakeAsset:
    def __init__(self, href):
        self.href = href

class FakeItem:
    def __init__(self, id, datetime, assets):
        self.id = id
        self.datetime = datetime
        self.assets = assets

@pytest.fixture
def mock_stac_item():
    return FakeItem(
        id="test_tile",
        datetime=datetime(2023, 6, 1),
        assets={
            "B01": FakeAsset("http://fake.com/B01.tif"),
            "B02": FakeAsset("http://fake.com/B02.tif"),
            "mask": FakeAsset("http://fake.com/mask.tif"),
            "meta": FakeAsset("http://fake.com/metadata.xml"),
        },
    )

@pytest.fixture
def mock_raster_data():
    return np.array([[1, 1], [1, 1]], dtype=np.uint8), {"nodata": 0, "driver": "GTiff"}, (0, 0, 2, 2)

@pytest.fixture
def patch_all(mock_raster_data):
    with patch("stac_downloader.downloading.download_file",), \
         patch("stac_downloader.downloading.download_raster_file", return_value=mock_raster_data), \
         patch("stac_downloader.raster_processing.resample_raster", side_effect=lambda a, b, c, d, e: (a, b)), \
         patch("stac_downloader.raster_processing.save_band",), \
         patch("stac_downloader.raster_processing.build_bandstacked_vrt", side_effect=lambda path, bp, bn: path):

        yield

def fake_mask_hook(bands, resolution):
    combined_mask = np.ones((2, 2), dtype=np.uint8)
    return {"nodata": 0}, combined_mask

def fake_post_hook(item, band_paths, band_names, mask, file_asset_paths, resolution, output_folder):
    band_paths["hooked"] = "hook_output.tif"
    band_names.append("hooked")
    return band_paths, band_names

def test_stac_downloader_integration(tmp_path, mock_stac_item, patch_all):
    from stac_downloader.stac_downloader import STACDownloader

    downloader = STACDownloader()
    downloader.register_masking_hook(fake_mask_hook, ["mask"])
    downloader.register_postdownload_hook(fake_post_hook)

    output = downloader.download_items(
        items=[mock_stac_item],
        raster_assets=["B01", "B02"],
        file_assets=["meta"],
        mask_assets=["mask"],
        output_folder=str(tmp_path),
        overwrite=True,
        resolution=10,
        resampling_method="nearest",
        save_mask_as_band=True,
        num_workers=1,
    )

    assert len(output) == 1
    vrt_path, file_asset_paths, band_paths, band_names_ordered = output[0]
    assert vrt_path.endswith(".vrt")
    assert "meta" in file_asset_paths
    assert isinstance(file_asset_paths["meta"], str)

    assert len(band_paths) == 4
    assert band_names_ordered == ["B01", "B02", "mask", "hooked"]

    expected_output_contains = ["test_tile_B01_10m.tif", "test_tile_B02_10m.tif", "test_tile_mask_10m.tif", "hook_output.tif"]

    out_fnames = [v.split("/")[-1] for v in band_paths.values()]

    for fname in expected_output_contains:
        assert fname in out_fnames
