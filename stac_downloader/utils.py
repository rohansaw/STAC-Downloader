import logging
import shutil
import subprocess

import colorlog
import time

import geopandas as gpd
from shapely import make_valid


def get_logger():
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",  # %(module)-15s
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "gray",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger = colorlog.getLogger("logger")

    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

def run_subprocess(cmd: list, step_desc: str, logger):
    """
    Execute a subprocess and stream its output to the console.
    If it fails, log error and raise.
    """
    logger.info(f"Starting: {step_desc}\n  Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        # Inherit stdout/stderr so user sees real-time output
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_desc}: return code {e.returncode}")
        raise RuntimeError(f"{step_desc} failed (exit code {e.returncode})")
    logger.info(f"Completed: {step_desc} in {time.time() - t0:.2f} seconds")

def persist_file(temp_path, target_path):
    shutil.move(temp_path, target_path)

def prepare_geometry(gdf: gpd.GeoDataFrame, enveloped=False):
    """Prepares a geodataframe for a STAC API request,
    by projecting to EPSG:4326, making valid and creating union.

    If enveloped = True, will create a bounding box for every geometry, 
    which helps in reducing size for smaller requests.
    """
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    geoms = gdf.geometry

    # Create bounding box for every polygons to size
    if enveloped:
        geoms = geoms.envelope

    geometry = geoms.union_all()

    return geometry

