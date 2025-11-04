# STAC-Downloader
Simple Utility for efficient parallelized downloading, resampling and adjusting of STAC data. Designed to preprocess remotely sensed data per tile parallelized to create inference ready tiles.

## Installation
```bash
# Install the package with pip
pip install git+https://github.com/rohansaw/STAC-Downloader
```

## Usage
First identify your STAC catalog's URL and examine which assets you are interested in. To download these assets, the `STAC Downloader` provides two options:

**Option 1: Command Line Interface (CLI)**

The CLI provides a download utility for STAC data with __reduced functionality__. In contrast to the python module, it does not allow directly appplying a mask nor setting custom functions to process the downloaded imagery directly. Also it will download all bands as int16 and use the same resampling method per band. However, it might come in handy if you just want to download metadata and imagery and resample imagery to a common resolution. 

IMPORTANT NOTE: If not specified differently all bands are currently saved as int16! Use the Option 2 to specify band dtypes. Also be careful when using bilinear resampling as this will be applied to e.g a mask band aswell.
It is recommended to use the python module instead of the basic cli currently.

```bash
Usage: stac_downloader [OPTIONS]

  Download Imagery and Metadata from a STAC catalog and resample.

Options:
  --catalog-url TEXT              URL of the STAC catalog  [default:
                                  https://earth-search.aws.element84.com/v1]
  --collection-name TEXT          STAC collection name  [default:
                                  sentinel-2-c1-l2a]
  --raster-assets TEXT            Raster asset names  [default: green, red,
                                  rededge1, rededge2, rededge3, nir08, swir16,
                                  swir22]
  --file-assets TEXT              Non-raster file asset names  [default:
                                  granule_metadata]
  --max-cloud-cover INTEGER       Maximum cloud cover % to filter on. Only available
                                  if eo filter parameter is availble in STAC catalog.
                                  [default: 60]
  --geometry-path FILE            GeoJSON file with search geometry. Must be in EPSG:4326.
                                  [required]
  --start-date TEXT               Search start date (YYYY-MM-DD)  [required]
  --end-date TEXT                 Search end date (YYYY-MM-DD)  [required]
  --resolution INTEGER            Output resolution in meters  [default: 20]
  --resampling-method [NEAREST|BILINEAR]
                                  Resampling method  [default: NEAREST]
  --output-folder DIRECTORY       Directory to save downloaded scenes
                                  [required]
  --overwrite / --no-overwrite    Overwrite existing files  [default: no-
                                  overwrite]
  --num-workers INTEGER           Parallel download workers  [default: 1]
  --help                          Show this message and exit.
```

**Option 2: Import Package**

The `STAC Downloader` is designed to be generic for different `STAC` catalogs and processing options. Therefore, importing the package provides you with more flexibility for custom processing of the data in memory before materializing to disk. This is handled through the setup of custom `hooks`. A `hook` allows to specify a function that processes a singe (or multiple) bands in a specific way. There are a few differet types of hooks. The available hook attachment points are the following:
- `masking_hook`: Created a mask from `mask_bands` that will be used for masking all other raster bands. Your provided function, will preprocess the mask band into a binary mask.
- `band_processing_hoosk`: This allows processing bands individually with a provided function (e.g adding a offset or specific bands etc.).
- `post_download_hooks`: Allows running any form of e.g bandcombination, creation of additional bands etc after all raster bands have been downloaded. This is not a only in-memory operation, as it re-reads the previosly saved bands and rewrites data.

```python
from stac_downloader.stac_downloader import STACDownloader
from typing import Dict

# Initialize the downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL) # e.g catalog_url like  https://planetarycomputer.microsoft.com/api/stac/v1

# Optional: Define a masking hook, that receives dictionary of bands as an input
def masking_hook(mask_assets: Dict[str, np.ndarray]):
    mask = []

    for asset_name, raster in mask_assets.items():
        # build BINARY mask with 1 being pixels to keep
    
    return mask

# Optional: Define a postprocessing hook with the following parameters
def postprocessing_hook(item: pyStacItem,
    band_paths: Dict[str, str],
    band_names: List[str],
    mask: np.ndarray,
    file_asset_paths: Dict[str, str],
    resolution: float,
    output_folder: str):

    # read bands, combine bands and save new output to output folder etc.

    # Add new bands to bandpaths, remove old ones if necessary

    # Update band names to keep the correct order for band stacking

    return band_paths, band_names

# Register hooks
stac_downloader.register_masking_hook(masking_hook)
stac_downloader.register_postdownload_hook(postprocessing_hook)

# Query STAC catalog to fetch item. A simple interface is provided with .query_catalog
# If you require more functionality of the STAC catalog, you can refer to pyStacClient which also returns pyStacClient.Item
items = stac_downloader.query_catalog(
    collection_name=STAC_COLLECTION_NAME, # Name of the stac collection, e.g sentinel-2-l2a
    start_date=START_DATE, # YYYY-MM-DD
    end_date=END_DATE, # YYYY-MM-DD
    geometry=GEOMETRY, # E.g from geopandas. Must be in EPSG:4326 typically
    query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
)

# Download stac items - Checkout examples for actual parameters
downloaded_item_paths = stac_downloader.download_items(
    items=items,
    raster_assets=RASTER_ASSETS, # List[str]: Asset names of the STAC item to download as rasters. Are pre-processable/reprojectable.
    file_assets=FILE_ASSETS, # List[str]: Asset names of the STAC item to download as the original files. No pre-processing/reprojection applied.
    mask_assets=MASK_ASSETS, # List[str]: Asset names of the STAC item that are used to build a mask (e.g SCL)
    output_folder=OUTPUT_FOLDER, # str: Where to save all output images.
    overwrite=OVERWRITE, # bool: If set to true will overwrite already existing imagery in output_folder
    resolution=RESOLUTION, # Resolution to resample to in CRS of the data. 
    resampling_spec=RESAMPLING_METHOD, # Either nearest | bilinear or Dict[str, nearest|bilinear] with the key being the bandname.
    num_workers=NUM_WORKERS, # Number of parallel download jobs
    raster_asset_target_dtypes=RASTER_ASSET_TARGET_DTYPES # Dict[str: Any]: The dtype in which to save each raster asset (e.g {'B4': int16}). Can only create vrt if all are the same. If not specified will use int16! To specify a dtype for the Maskband use "mask" as the key! Caution if not saving the band in a dtype that is not it's original dtype as this might cause values to overflow.
    build_vrt=True
)

```

Head over to the examples section (`stac_downloader/examples`) to see how to fetch & process Sentinel-2 imagery, with sun angles and masking. If you want to explore the examples, make sure to install the additional requirements under `stac_downloader/examples/requirements.txt`.
