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

The CLI provides a download utility for STAC data with __reduced functionality__. It does not allow directly appplying a mask nor setting custom functions to process the downloaded imagery directly. However, it might come in handy if you just want to download metadata and imagery and resample imagery to a common resolution.

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
  --geometry-path FILE            GeoJSON file with search geometry
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

The `STAC Downloader` is designed to be generic for different `STAC` catalogs and processing options. Therefore, importing the package provides you with more flexibility for custom processing through the setup of custom hooks. The available hook attachment points are the following:
- `masking_hook`: Created a mask from `mask_bands` that will be used for masking all other raster bands.
- `band_processing_hoosk`: This allows processing bands individually with a provided function (e.g adding a offset or specific bands etc.).
- `post_download_hooks`: Allows running any form of e.g bandcombination, creation of additional bands etc after all raster bands have been downloaded .

```python
from stac_downloader.stac_downloader import STACDownloader
from typing import Dict

# Initialize the downloader
stac_downloader = STACDownloader(catalog_url=STACK_CATALOG_URL)

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

# Query STAC catalog to fetch items - Checkout examples for actual parameters
items = stac_downloader.query_catalog(
    collection_name=STAC_COLLECTION_NAME,
    start_date=START_DATE,
    end_date=END_DATE,
    geometry=GEOMETRY,
    query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
)

# Download items - Checkout examples for actual parameters
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

```

Head over to the examples section (`stac_downloader/examples`) to see how to fetch & process Sentinel-2 imagery, with sun angles and masking.
