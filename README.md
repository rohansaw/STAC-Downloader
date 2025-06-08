# STAC-Downloader
Simple Utility for efficient parallelized downloading, resampling and adjusting of STAC data. Designed to preprocess remotely sensed data per tile parallelized to create inference ready tiles.

## Installation
```bash
# Install the package with pip
pip install git+https://github.com/rohansaw/STAC-Downloader
```

## Usage

**Option 1: Command Line Interface (CLI)**
The CLI provides a download utility for STAC data with **reduced functionality**. It does not allow directly appplying a mask nor setting custom functions to process the downloaded imagery directly. However, it might come in handy if you just want to download metadata and imagery and resample imagery to a common resolution.

```bash
python
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

**Option 2: Import Package**
The `STAC Downloader` is designed to be generic for different `STAC` catalogs and processing options. Therefore, import the package provides you with more flexibility for custom processing through the setup of custom hooks. The availble hook attachment points  are the following:
- `masking_hook`: Created a mask from `mask_bands` that will be used for masking all other raster bands.
- `band_processing_hoosk`: Not yet implemented. This will allow processing bands individually with the provided function.
- `post_download_hooks`: Allows running any form of e.g bandcombination, creation of additional bands etc after all raster bands have been downloaded .

```python
# TODO: Need to add an example on how to import and use the most relevant functions
```

Head over to the examples section (`stac_downloader/examples`) to see how to fetch & process Sentinel-2 imagery, with sun angles and masking.