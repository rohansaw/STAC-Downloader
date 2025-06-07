from enum import Enum
import os
import subprocess
import numpy as np
from rasterio.warp import Resampling, calculate_default_transform, reproject
import rasterio as rio

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


def resample_raster(raster, profile: dict, src_bounds, target_resolution: float, resampling_method: ResamplingMethod):
    src_resolution = profile['resolution']
    src_crs = profile['crs']
    src_width = profile['width']
    src_height = profile['height']
    src_dtype = profile['dtype']
    src_transform = profile['transform']

    rio_resampling_method = Resampling.from_string(resampling_method.value)


    if src_resolution != (target_resolution, target_resolution):
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, src_crs, src_width, src_height, *src_bounds, resolution=target_resolution
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

    with rio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, band_name)

    return output_path


def build_bandstacked_vrt(output_file_path: str, band_paths: dict, band_names: list[str], create_gtiff=False, blocksize=256):
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
        raise RuntimeError(f"gdalbuildvrt failed (exit {e.returncode}): {e.stderr.decode()}") from e
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
        raise FileNotFoundError(f"Output GeoTIFF file was not created: {output_file_path_gtiff}")

    return output_file_path