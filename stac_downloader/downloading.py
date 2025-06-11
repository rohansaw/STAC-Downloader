import os

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


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

        # Check if filesize matches expected size
        expected_size = int(response.headers.get("Content-Length", 0))
        if expected_size > 0 and os.path.getsize(output_path) != expected_size:
            raise IOError(
                f"Downloaded file size {os.path.getsize(output_path)} does not match expected size {expected_size}."
            )
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e
