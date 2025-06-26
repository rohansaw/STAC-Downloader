import os
from pathlib import Path
import subprocess
from enum import Enum
import tempfile

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tenacity import retry, stop_after_attempt, wait_exponential

from stac_downloader.utils import persist_file


# Non library specific resampling method enum
class ResamplingMethod(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"

    @classmethod
    def from_string(cls, method_str):
        try:
            return cls[method_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid resampling method: {method_str}")


# Using retry since we are sometimes reading from remote filesystems (e.g. S3) and they can be flaky.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def resample_raster(
    raster_path: str, target_resolution: float, resampling_method: ResamplingMethod
):
    with rio.open(raster_path) as src:
        if src.count > 1:
            raise ValueError(
                "The input file has more than one band. Currently only handling single band rasters."
            )

        raster = src.read(1)  # Read the first band
        src_bounds = src.bounds
        src_resolution = src.res
        src_width = src.width
        src_height = src.height
        src_crs = src.crs
        src_dtype = src.dtypes[0]
        src_transform = src.transform
        profile = src.profile.copy()

    if resampling_method == ResamplingMethod.NEAREST:
        rio_resampling_method = Resampling.nearest
    elif resampling_method == ResamplingMethod.BILINEAR:
        rio_resampling_method = Resampling.bilinear
    else:
        raise ValueError(f"Unsupported resampling method: {resampling_method}")

    if src_resolution != (target_resolution, target_resolution):
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            src_crs,
            src_width,
            src_height,
            *src_bounds,
            resolution=target_resolution,
        )

        resampled_raster = np.empty((dst_height, dst_width), dtype=src_dtype)
        reproject(
            source=raster,
            destination=resampled_raster,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=rio_resampling_method,
            num_threads=4,
        )
        profile.update(
            {
                "height": resampled_raster.shape[0],
                "width": resampled_raster.shape[1],
                "transform": dst_transform,
            }
        )
        return resampled_raster, profile
    else:
        return raster, profile


def save_band(raster, profile, output_path, band_name):
    profile.update(
        {
            "driver": "GTiff",
            "compress": "LZW",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "dtype": np.int16,  # Forcing to Int16 for now
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_fname = os.path.join(tmp_dir, Path(output_path).name)
        with rio.open(
            temp_fname,
            "w",
            **profile,
        ) as dst:
            dst.write(raster, 1)
            dst.set_band_description(1, band_name)    
    
        persist_file(temp_fname, output_path)

    return output_path


def build_bandstacked_vrt(
    output_file_path: str,
    band_paths: dict,
    band_names: list[str],
    create_gtiff=False,
    blocksize=256,
):
    # Use order of band_names to create the VRT
    band_paths_ordered = [band_paths[band_name] for band_name in band_names]

    # TODO use blocksize also for vrt if gdal is newer than (not sure which one need to check)
    vrt_build_command = [
        "gdalbuildvrt",
        "-separate",
        "-overwrite",
        output_file_path,
        *band_paths_ordered,
    ]

    try:
        subprocess.run(vrt_build_command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"gdalbuildvrt failed (exit {e.returncode}): {e.stderr.decode()}"
        ) from e
    except OSError as e:
        raise RuntimeError(f"Failed to launch gdalbuildvrt: {e}") from e

    if create_gtiff:
        # Create a GTiff from the VRT for easier exporting/debugging.
        output_file_path_gtiff = output_file_path.replace(".vrt", ".tif")
        geotiff_build_command = [
            "gdal_translate",
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "compress=LZW",
            "-co",
            f"blocksize={blocksize}",
            output_file_path,
            output_file_path_gtiff,
        ]
        try:
            subprocess.run(geotiff_build_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"gdal_translate failed (exit {e.returncode}): {e.stderr.decode()}"
            ) from e
        except OSError as e:
            raise RuntimeError(f"Failed to launch gdal_translate: {e}") from e

    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"Output VRT file was not created: {output_file_path}")

    if create_gtiff and not os.path.exists(output_file_path_gtiff):
        raise FileNotFoundError(
            f"Output GeoTIFF file was not created: {output_file_path_gtiff}"
        )

    return output_file_path


def apply_mask(raster: np.ndarray, mask: np.ndarray, nodata_value):
    """
    Apply a mask to the raster data.
    The mask should be a boolean array where True indicates pixels to keep.
    """
    if raster.shape != mask.shape:
        raise ValueError("Raster and mask must have the same shape.")

    masked_raster = np.where(
        mask, raster, nodata_value
    )  # Replace masked pixels with nodataval
    return masked_raster


def is_binary(raster: np.ndarray):
    """
    Check if the raster is binary (contains only 0s and 1s).
    """
    unique_values = np.unique(raster)
    return (
        np.array_equal(unique_values, [0, 1])
        or np.array_equal(unique_values, [1])
        or np.array_equal(unique_values, [0])
    )
