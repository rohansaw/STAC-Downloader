# STAC-Downloader
Simple Utility for efficient parallelized downloading, resampling and adjusting of STAC data. Designed to preprocess remotely sensed data per tile parallelized to create inference ready tiles.

## Setup
```bash
# Clone the repository
git clone https://github.com/rohansaw/STAC-Downloader

# Install the package
```

## Usage

**Option 1: Command Line Interface (CLI)**
Not implemented yet.

**Option 2: Custom Wrapper**
The `STAC Downloader` is designed to be generic for different `STAC` catalogs and processing options. Therefore, you can import the package and also setup your custom hooks. The availble hooks are the following:
- `masking_hook`: Created a mask from `mask_bands` that will be used for masking all other raster bands.
- `band_processing_hoosk`: Not yet implemented. This will allow processing bands individually with the provided function.
- `post_download_hooks`: Allows running any form of e.g bandcombination, creation of additional bands etc after all raster bands have been downloaded .

```python
# TODO: Need to add an example on how to import and use the most relevant functions
```

Head over to the examples section (`stac_downloader/examples`) to see how to fetch & process Sentinel-2 imagery, with sun angles and masking.