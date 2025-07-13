from datetime import datetime
import os
import time

import geopandas as gpd
import pandas as pd
from sentinel2_hooks import build_geometry_band_adder, build_s2_masking_hook, s2_harmonization_processor
import planetary_computer

from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger, prepare_geometry

# ##################################################################################
# Sentiel-1 RTC Downloader Example.
# Concrete implementation of the STAC downloader for Sentinel-1 RTC data from 
# Microsoft Planetary Computer.
# ##################################################################################
# Downloads Sentinel-1 Radiometrically Terrain Corrected data from Microsofts Planetary Computer.
#
####################################################################################

logger = get_logger()

# Define MPC-Specific Params
STACK_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION_NAME = "sentinel-1-rtc"
RASTER_ASSETS = ["hh", "hv", "vh","vv"]
GEOMETRY_PATH = "/home/rohan/nasa-harvest/vercye/data/Ukraine/poltava_hull.geojson"
RESOLUTION = 10  # Resolution in meters
START_DATE = "2023-01-01"
END_DATE = "2023-01-05"
OUTPUT_FOLDER = "/home/rohan/Downloads/sentinel1_downloads"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 16  # Number of parallel workers for downloading

MODIFIER = planetary_computer.sign # Required from MPC

# Setup STAC Downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL, logger=logger, stac_item_modifier=MODIFIER)

# Query the STAC catalog
logger.info(f"Searching for items from {START_DATE} to {END_DATE}...")
t0 = time.time()

gdf = gpd.read_file(GEOMETRY_PATH)

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

print(items[0].assets.keys())

logger.info(f"Found {len(items)} items")
logger.info(f"Search took {time.time() - t0:.2f} seconds")

# Download the items
logger.info("Starting download of items...")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
downloaded_item_paths = stac_downloader.download_items(
    items=items,
    raster_assets=RASTER_ASSETS,
    file_assets=[],
    mask_assets=[],
    output_folder=OUTPUT_FOLDER,
    overwrite=OVERWRITE,
    resolution=RESOLUTION,
    resampling_method=RESAMPLING_METHOD,
    num_workers=NUM_WORKERS,
)

logger.info(f"Downloads saved under {OUTPUT_FOLDER}.")
