import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from stac_downloader.utils import persist_file


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

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_fname = os.path.join(tmp_dir, Path(output_path).name)
            with open(temp_fname, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Check if filesize matches expected size
            expected_size = int(response.headers.get("Content-Length", 0))
            if expected_size > 0 and os.path.getsize(temp_fname) != expected_size:
                raise IOError(
                    f"Downloaded file size {os.path.getsize(temp_fname)} does not match expected size {expected_size}."
                )
            
            # Validate xml not corrupted during download
            if os.path.splitext(temp_fname)[1] == '.xml':
                try:
                    with open(temp_fname, "r") as f:
                        xml_data = f.read()
                    xml_root = ET.fromstring(xml_data)
                except Exception as e:
                    raise Exception('Corrupted XML File detected.')

            persist_file(temp_fname, output_path)
