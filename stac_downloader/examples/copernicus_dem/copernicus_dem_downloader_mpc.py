import os
import time

import geopandas as gpd
import numpy as np
import planetary_computer

from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger, prepare_geometry

# --------------------------------------------------------------------------------------------------
# Copernicus DEM Downloader Example
# This script demonstrates how to download Copernicus GLO-30 Digital Elevation Model (DEM) data
# from the Microsoft Planetary Computer (MPC) using the `stac_downloader` utility.
#
# Dataset: "cop-dem-glo-30" — 30-meter resolution global DEM
# Source: Microsoft Planetary Computer (STAC API)
# Output: Raw elevation data in GeoTIFF format (e.g., "data" asset). Currently no resizing is supported
# --------------------------------------------------------------------------------------------------


logger = get_logger()

# Define MPC-Specific Params
STACK_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION_NAME = "cop-dem-glo-30"
FILE_ASSETS = ["data"]
RASTER_ASSETS=[]
GEOMETRY_PATH = "/home/rohan/nasa-harvest/vercye/data/Ukraine/poltava_hull.geojson"
# USING FILE ASSET ONLY - NO RESIZING RESOLUTION = 30  # Resolution in meters
RESOLUTION = None
OUTPUT_FOLDER = "/home/rohan/Downloads/copernicus_dem"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 1  # Number of parallel workers for downloading

RASTER_BAND_DTYPES = {
    "data": np.float32,
}

MODIFIER = planetary_computer.sign # Required from MPC

# Setup STAC Downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL, logger=logger, stac_item_modifier=MODIFIER)

# Query the STAC catalog
logger.info(f"Searching for dem items...")
t0 = time.time()

gdf = gpd.read_file(GEOMETRY_PATH)

try:
    geometry = prepare_geometry(gdf)
    items = stac_downloader.query_catalog(
        collection_name=STAC_COLLECTION_NAME,
        geometry=geometry,
    )
except Exception as e:
    # Try using bounding boxes instead of exact polygon to reduce size.
    # This is a typical reason why the requests fails.
    geometry = prepare_geometry(gdf, enveloped=True)
    items = stac_downloader.query_catalog(
        collection_name=STAC_COLLECTION_NAME,
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
    file_assets=FILE_ASSETS,
    mask_assets=[],
    output_folder=OUTPUT_FOLDER,
    overwrite=OVERWRITE,
    resolution=RESOLUTION,
    resampling_method=RESAMPLING_METHOD,
    num_workers=NUM_WORKERS,
    raster_asset_target_dtypes=RASTER_BAND_DTYPES
)

logger.info(f"Downloads saved under {OUTPUT_FOLDER}.")
