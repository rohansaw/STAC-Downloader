import os
import time

import geopandas as gpd
from sentinel2_hooks import add_geometry_bands, build_s2_masking_hook

from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger

# ##################################################################################
# Sentiel-2 Downloader Example.
# Concrete implementation of the STAC downloader for Sentinel-2 data.
# ##################################################################################
# Downloads Sentinel-2 Level 2A Collection 1 data from the AWS STAC catalog.
# ATTENTION: This data has all be processed with baseline >= 5.0.
# Therefore all data has a shift of 1000 in the reflectance bands.
#
# This example shows how to download Sentinel-2 data from the AWS STAC catalog.
# It applies a mask based on the SCL layer and cloud/snow masks from S2 Cloudless.
# This example also shows how to apply custom bands from geometry angles in metdata
####################################################################################


logger = get_logger()

STACK_CATALOG_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION_NAME = "sentinel-2-c1-l2a"
MASK_ASSETS = ["scl", "cloud", "snow"]

RASTER_ASSETS = [
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "swir16",
    "swir22",
]
FILE_ASSETS = [
    "granule_metadata",
]
MAX_CLOUD_COVER = 60  # Maximum cloud cover percentage to filter items
GEOMETRY_PATH = "/home/rohan/nasa-harvest/vercye/data/Ukraine/poltava_hull.geojson"
RESOLUTION = 20  # Resolution in meters
START_DATE = "2023-01-01"
END_DATE = "2023-01-05"
OUTPUT_FOLDER = "/home/rohan/Downloads/sentinel2_c1_l2a_downloads"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 16  # Number of parallel workers for downloading

CLOUD_THRESH = 10  # Threshold for probability cloud mask in percent
SNOWPROB_THRESH = 15  # Threshold for snow probability mask in percent
SCL_KEEP_CLASSES = [4, 5]

# Setup STAC Downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL, logger=logger)

# Register masking hook based on SCL & Cloud Probs with a threshold for cloud and snow
s2_masking_hook = build_s2_masking_hook(
    cloud_thresh=CLOUD_THRESH,
    snowprob_thresh=SNOWPROB_THRESH,
    scl_keep_classes=SCL_KEEP_CLASSES,
)
stac_downloader.register_masking_hook(s2_masking_hook)

# Register geometry bands hook to create bands adding cosines of angles from metadata
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

logger.info(f"Downloads saved under {OUTPUT_FOLDER}.")
