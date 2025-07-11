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
END_DATE = "2023-01-05"
OUTPUT_FOLDER = "/home/rohan/Downloads/sentinel2_mpc_downloads"
OVERWRITE = False  # Set to True to overwrite existing files
RESAMPLING_METHOD = ResamplingMethod.NEAREST  # Resampling method for raster assets
NUM_WORKERS = 16  # Number of parallel workers for downloading

SCL_KEEP_CLASSES = [4, 5] # Class Values that should be kept from SCL Mask. All others are set to nodata
SCL_BANDNAME = 'SCL' # SCL Bandname

MODIFIER = planetary_computer.sign_inplace # Required from MPC

# Define deduplication helper
def dedupliate(items):
    item_entries = [
        {
            'mgrs_tile_id': item.id.split('_')[4],
            'date': datetime.strptime(item.id.split('_')[2], "%Y%m%dT%H%M%S").date(),
            'processing_date':  datetime.strptime(item.id.split('_')[5], "%Y%m%dT%H%M%S"),
            'item': item,
            'invalid_formatting': len(item.id.split('_')[3]) != 4 or item.id.split('_')[3][0] != 'R'
        } for item in items
    ]

    df = pd.DataFrame(item_entries)

    if df['invalid_formatting'].any():
        first_invalid = df[df['invalid_formatting'] == True].iloc[0]
        raise Exception(f"Can't deduplicate due to unexpected item id formatting. Expecting id like S2B_MSIL2A_20230413T105619_R094_T31UDP_20240829T164929. Got {first_invalid['item'].id}.")

    # Group by tile_id and acuqisition data. Sort by processing date within group and only keep newest.
    df_grouped = df.sort_values("processing_date").groupby(['mgrs_tile_id', 'date']).tail(1)

    return df_grouped['item'].to_list()

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

# Register hook to harmonize the data processed with baseline before 4, to match the outputs from baseline 4
stac_downloader.register_bandprocessing_hook(s2_harmonization_processor, band_assets=RASTER_ASSETS)

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
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
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
