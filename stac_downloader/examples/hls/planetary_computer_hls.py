import os
import time
from typing import Dict, List

import geopandas as gpd
import numpy as np
import rasterio as rio
import planetary_computer
from pystac.item import Item as pyStacItem


from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger, prepare_geometry

# ##################################################################################
# Harmonized Landsat Sentinel-2 Downloader Example.
# Concrete implementation of the STAC downloader for HLS data from 
# Microsoft Planetary Computer.
# ##################################################################################
# Downloads Harmoniyzed Landsat Sentinel-2 Imagery from Microsoft Planetary Computer
# Specify whether to use the S30 or L30 version below.
####################################################################################

logger = get_logger()

# Define params
COLLECTION_TYPE = "L30" # S30 for Sentinel, L30 for Landsat
RASTER_ASSETS = ["VZA", "SZA", "SAA", "VAA", "B03", "B04", "B05", "B06", "B07", ]
GEOMETRY_PATH = "/home/rohan/nasa-harvest/vercye/data/Ukraine/poltava_hull.geojson" # For search
RESOLUTION = 30  # Resolution in meters
START_DATE = "2023-01-01"
END_DATE = "2023-01-05"
OUTPUT_FOLDER = "/home/rohan/Downloads/hls_debug"

MASK_ASSETS=["Fmask"]
STACK_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 1  # Number of parallel workers for downloading

# All other bands will use int16 as dtype.
# Typically this would not work to build a VRT from different dtypes per band.
# However we process these bands to cosAngles in int16 and remove the original bands.
RASTER_BAND_DTYPES = {
    "VZA": np.uint16,
    "SZA": np.uint16,
    "SAA": np.uint16,
    "VAA": np.uint16
}

if COLLECTION_TYPE == "S30":
    STAC_COLLECTION_NAME = "hls2-s30"
elif COLLECTION_TYPE == "L30":
     STAC_COLLECTION_NAME = "hls2-l30"


def build_snow_clouds_mask(maskbands):
    """Create a mask for pixels that are clouds, cloud shadow, adjacent to cloud(-shadow) or snow/ice.
    
    See Appendix A: https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
    """
    fmask_meta, fmask = maskbands["Fmask"]
    mask = np.ones_like(fmask)  # Start with all valid (1)

    is_cloud = ((fmask >> 1) & 1).astype(bool) # bit 1
    is_cloud_shadow = ((fmask >> 3) & 1).astype(bool) # bit 3
    is_adjacent = ((fmask >> 2) & 1).astype(bool) # bit 2
    is_snow = ((fmask >> 4) & 1).astype(bool) # bit 4

    should_be_masked = is_cloud | is_adjacent | is_snow | is_cloud_shadow
    mask = np.where(should_be_masked, 0, mask)

    return {}, mask

def create_geometry_bands(
    item: pyStacItem,
    band_paths: Dict[str, str],
    band_names: List[str],
    mask: np.ndarray,
    file_asset_paths: Dict[str, str],
    resolution: float,
    output_folder: str
):
    """Create bands containing cosVZA, cosSZA, cosRAA from present bands in HLS.
    Based on Leaf Toolbox in GEE (toolsL8 script).
    """

    def save_cos_band(cos_arr: np.ndarray, out_path: str, profile: dict):
        profile.update(dtype="uint16", count=1, compress="lzw")
        with rio.open(out_path, "w", **profile) as dst:
            dst.write(np.squeeze(cos_arr), 1)

    with rio.open(band_paths['VZA']) as src:
        vza = src.read()
        cos_vza = np.cos(np.deg2rad(vza /  100)) * 10000
        profile_vza = src.profile

    cos_vza_path = os.path.join(output_folder, f"{item.id}_cos_vza_{str(resolution)}m.tif")
    save_cos_band(cos_vza, cos_vza_path, profile_vza)

    with rio.open(band_paths['SZA']) as src:
        sza = src.read()
        cos_sza = np.cos(np.deg2rad(sza /  100)) * 10000
        profile_sza = src.profile

    cos_sza_path = os.path.join(output_folder, f"{item.id}_cos_sza_{str(resolution)}m.tif")
    save_cos_band(cos_sza, cos_sza_path, profile_sza)

    with rio.open(band_paths['VAA']) as src:
        vaa = src.read()
    
    with rio.open(band_paths['SAA']) as src:
        saa = src.read()
        profile_saa = src.profile

    raa = vaa - saa
    cos_raa =  np.cos(np.deg2rad(raa / 100)) * 10000
    cos_raa_path = os.path.join(output_folder, f"{item.id}_cos_raa_{str(resolution)}m.tif")
    save_cos_band(cos_raa, cos_raa_path, profile_saa)

    # Ensure correct bandorder for output
    band_paths["cos_vza"] = cos_vza_path
    band_paths["cos_sza"] = cos_sza_path
    band_paths["cos_raa"] = cos_raa_path

    # Drop original VZA SZA VAA SAA bands and instead add new bands
    band_names = ["cos_vza", "cos_sza", "cos_raa"] + band_names[4:]

    return band_paths, band_names


MODIFIER = planetary_computer.sign # Required from MPC

# Setup STAC Downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL, logger=logger, stac_item_modifier=MODIFIER, raster_asset_target_dtypes=RASTER_BAND_DTYPES)
stac_downloader.register_masking_hook(build_snow_clouds_mask)
stac_downloader.register_postdownload_hook(create_geometry_bands)

# Query the STAC catalog
logger.info(f"Searching for items from {START_DATE} to {END_DATE}...")
t0 = time.time()

gdf = gpd.read_file(GEOMETRY_PATH)

# Apply Cloudmask

try:
    geometry = prepare_geometry(gdf)
    items = stac_downloader.query_catalog(
        collection_name=STAC_COLLECTION_NAME,
        start_date=START_DATE,
        end_date=END_DATE,
        geometry=geometry,
    )
except Exception as e:
    # Try using bounding boxes instead of exact polygon to reduce size.
    # This is a typical reason why the requests fails.
    geometry = prepare_geometry(gdf, enveloped=True)
    items = stac_downloader.query_catalog(
        collection_name=STAC_COLLECTION_NAME,
        start_date=START_DATE,
        end_date=END_DATE,
        geometry=geometry,
    )

logger.info(f"Found {len(items)} items")
logger.info(f"Search took {time.time() - t0:.2f} seconds")

# Download the items
logger.info("Starting download of items...")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
downloaded_item_paths = stac_downloader.download_items(
    items=items,
    raster_assets=RASTER_ASSETS,
    file_assets=[],
    mask_assets=MASK_ASSETS,
    output_folder=OUTPUT_FOLDER,
    overwrite=OVERWRITE,
    resolution=RESOLUTION,
    resampling_method=RESAMPLING_METHOD,
    num_workers=NUM_WORKERS,
)

logger.info(f"Downloads saved under {OUTPUT_FOLDER}.")
