import numpy as np
from rasterio import Env
import requests
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import rasterio as rio


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def download_file(url: str, output_path: str, overwrite: bool = True) -> None:
    """
    Download any file from a URL with retry logic.
    Should typically be used for downloading the metadata files not raster filer.
    Args:
        url (str): The URL to download the file from.
        output_path (str): The local path where the file should be saved.
        overwrite (bool): Whether to overwrite the file if it already exists.
    Raises:
        requests.RequestException: If there is an error during the request.
        IOError: If there is an error writing the file.
    """
    if not overwrite and os.path.exists(output_path):
        return

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    except (requests.RequestException, IOError) as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e
    

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def download_raster_file(url: str) -> tuple[np.ndarray, dict, tuple]:
    """
    Download a raster file at from a URL into memory with retry logic.
    Args:
        url (str): The URL to download the raster file from.
    """

    with Env(AWS_NO_SIGN_REQUEST="YES",
             GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
             GDAL_HTTP_MULTIRANGE="YES",
             GDAL_ENABLE_CURL_MULTI="YES",
             GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
             GDAL_HTTP_MULTIPLEX="YES",
             GDAL_HTTP_VERSION="2"
            ):
        with rio.open(url) as src:
            if src.count > 1:
                raise ValueError(
                    "The input file has more than one band. Currently only handling single band rasters."
                )

            data = src.read(1)  # Read the first band
            profile = src.profile
            bounds = src.bounds

            return data, profile, bounds
