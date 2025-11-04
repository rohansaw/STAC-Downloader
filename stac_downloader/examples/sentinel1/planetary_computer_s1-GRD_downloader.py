from collections import defaultdict
import os
import time

import geopandas as gpd
import numpy as np
import planetary_computer
from pyproj import CRS
from shapely.geometry import box

from pystac.item import Item as pyStacItem
from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger, prepare_geometry

# ##################################################################################
# Sentiel-1 RTC Downloader Example.
# Concrete implementation of the STAC downloader for Sentinel-1 RTC data from 
# Microsoft Planetary Computer.
# ##################################################################################
# Downloads Sentinel-1 GRD data from Microsofts Planetary Computer.
# This data is not yet sigma0 or gamma0 backscatter and will have to be processed with
# calibration data, e.g in the SNAP Toolbox.
####################################################################################

logger = get_logger()

raise NotImplementedError("This is not yet completely implemented")

# Define MPC-Specific Params
STACK_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION_NAME = "sentinel-1-grd"
RASTER_ASSETS = ["vv", "vh"]
FILE_ASSETS = ["schema-calibration-vv", "schema-calibration-vh"]
GEOMETRY_PATH = "/home/rohan/nasa-harvest/vercye/data/Ukraine/poltava_hull.geojson"
START_DATE = "2023-01-01"
END_DATE = "2023-01-05"
OUTPUT_FOLDER = "/home/rohan/Downloads/sentinel1_downloads"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 1  # Number of parallel workers for 
RESOLUTION = 10 # In the CRS of the most frequent tile!!
TARGET_CRS = None # Will be automatically determined from most frequent CRS of all scenes if not set

RASTER_BAND_DTYPES = {
    "vv": np.float32,
    "vh": np.float32,
}
MODIFIER = planetary_computer.sign # Required from MPC

# Setup STAC Downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL, logger=logger, stac_item_modifier=MODIFIER)

# Query the STAC catalog
logger.info(f"Searching for items from {START_DATE} to {END_DATE}...")
t0 = time.time()

gdf = gpd.read_file(GEOMETRY_PATH)

def to_db(raster: np.ndarray, raster_profile: dict, item: pyStacItem):
    raster = 10 * np.log10(raster)
    return raster, raster_profile, item

stac_downloader.register_bandprocessing_hook(to_db, ["vv", "vh"])

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

if not CRS:
    logger.info("Determining CRS to use (majority CRS of all scenes).")
    all_utms = defaultdict(0)
    for item in items:
        item_bbox = item.properties['proj:bbox']
        geom = box(*item_bbox)
        centroid = geom.centroid
        utm_crs = CRS.from_user_input(CRS.from_epsg(4326).utm_zone(centroid.x, centroid.y))
        all_utms[utm_crs] += 1
    
    majority_utm = max(all_utms, key=all_utms.get)
    logger.info(f"Using CRS: {majority_utm}")
    TARGET_CRS = majority_utm

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
    resampling_spec=RESAMPLING_METHOD,
    num_workers=NUM_WORKERS,
    target_crs=TARGET_CRS,
    raster_asset_target_dtypes=RASTER_BAND_DTYPES
)

logger.info(f"Downloads saved under {OUTPUT_FOLDER}.")
