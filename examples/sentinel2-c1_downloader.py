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

import pystac_client
import time

from stac_downloader import STACDownloader
from utils import get_logger
logger = get_logger()

from examples.sentinel2_hooks import sentinel2_geometry_bands_processor, build_s2_masking_hook


STACK_CATALOG_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION_NAME  = "sentinel-2-c1-l2a"
MASK_BANDS = ["scl", "cloud", "snow"]
MAX_CLOUD_COVER = 60  # Maximum cloud cover percentage to filter items

START_DATE = "2023-01-01"
END_DATE = "2023-01-31"


# Query the stac catalog to retrive items fulfilling the criteria
catalog = pystac_client.Client.open(STACK_CATALOG_URL)
collection = catalog.get_collection(STAC_COLLECTION_NAME)

query = {"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
geometry = None

# Search for STAC-items in the STAC catalog
logger.info(f"Searching for items from {START_DATE} to {END_DATE}...")
t0 = time.time()
search = catalog.search(
    collections=[STAC_COLLECTION_NAME],
    intersects=geometry,
    datetime=f"{START_DATE}/{END_DATE}",
    query=query,
)

items = search.item_collection()
logger.info(f"Found {len(items)} items")
logger.info(f"Search took {time.time() - t0:.2f} seconds")

# Use STAC Downloader to download the items
stac_downloader = STACDownloader(logger=logger)

# Register masking hook based on SCL & S2-Cloudless with a threshold for cloud and snow
s2_masking_hook = build_s2_masking_hook()
stac_downloader.register_masking_hook(s2_masking_hook)

# Register geometry bands hook to create bands adding cosines of angles from metadata
geometry_bands_hook = sentinel2_geometry_bands_processor
stac_downloader.register_postdownload_hook(geometry_bands_hook)