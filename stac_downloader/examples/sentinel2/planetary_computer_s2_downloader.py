import os
import time

import geopandas as gpd
from sentinel2_hooks import build_geometry_band_adder, build_s2_masking_hook
import planetary_computer

from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger

# ##################################################################################
# Sentiel-2 Downloader Example.
# Concrete implementation of the STAC downloader for Sentinel-2 data from 
# Microsoft Planetary Computer.
# ##################################################################################
# Downloads Sentinel-2 Level 2A data from Microsofts Planetary Computer.
# ATTENTION: Data will be harmonized to Processing Baseline 5. This means that all
# data before 25.01.2022 will be shifted by +1000 to match the Baseline 5 data.
#
# This example applies the SCL layer for cloud masking and shows how to apply
# custom bandwise processing functions
####################################################################################


logger = get_logger()

STACK_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION_NAME = "sentinel-2-l2a"
MASK_ASSETS = ["SCL"]

RASTER_ASSETS = [
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B8A",
                "B11",
                "B12",
]
FILE_ASSETS = [
    "granule-metadata",
]
MAX_CLOUD_COVER = 60  # Maximum cloud cover percentage to filter items
GEOMETRY_PATH = "/home/rohan/nasa-harvest/vercye/data/Ukraine/poltava_hull.geojson"
RESOLUTION = 20  # Resolution in meters
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"
OUTPUT_FOLDER = "/home/rohan/Downloads/sentinel2_c1_l2a_downloads"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 16  # Number of parallel workers for downloading

SCL_KEEP_CLASSES = [4, 5]
SCL_BANDNAME = 'SCL'

MODIFIER = planetary_computer.sign_inplace

# Setup STAC Downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL, logger=logger, stac_catalog_modifier=MODIFIER)

# Register masking hook based on SCL & Cloud Probs with a threshold for cloud and snow
s2_masking_hook = build_s2_masking_hook(
    scl_keep_classes=SCL_KEEP_CLASSES,
    scl_bandname=SCL_BANDNAME
)
stac_downloader.register_masking_hook(s2_masking_hook)

# Register geometry bands hook to create bands adding cosines of angles from metadata
add_geometry_bands = build_geometry_band_adder(granule_mtd_asset_name='granule-metadata')
stac_downloader.register_postdownload_hook(add_geometry_bands)

logger.info(f"Searching for items from {START_DATE} to {END_DATE}...")
t0 = time.time()
geometry = gpd.read_file(GEOMETRY_PATH).geometry.values[0] if GEOMETRY_PATH else None
items = stac_downloader.query_catalog(
    collection_name=STAC_COLLECTION_NAME,
    start_date=START_DATE,
    end_date=END_DATE,
    geometry=geometry,
    query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
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
    mask_assets=MASK_ASSETS,
    output_folder=OUTPUT_FOLDER,
    overwrite=OVERWRITE,
    resolution=RESOLUTION,
    resampling_method=RESAMPLING_METHOD,
    num_workers=NUM_WORKERS,
)

logger.info(f"Downloaded {len(downloaded_item_paths)} items to {OUTPUT_FOLDER}")
