#!/usr/bin/env python3
import os
import time

import click
import geopandas as gpd

from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger


@click.command(context_settings={"show_default": True})
@click.option(
    "--catalog-url",
    default="https://earth-search.aws.element84.com/v1",
    help="URL of the STAC catalog",
)
@click.option(
    "--collection-name",
    default="sentinel-2-c1-l2a",
    help="STAC collection name",
)
@click.option(
    "--raster-assets",
    multiple=True,
    default=[
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir08",
        "swir16",
        "swir22",
    ],
    help="Raster asset names",
)
@click.option(
    "--file-assets",
    multiple=True,
    default=["granule_metadata"],
    help="Non-raster file asset names",
)
@click.option(
    "--max-cloud-cover",
    default=60,
    show_default=True,
    help="Maximum cloud cover % to filter on.",
)
@click.option(
    "--geometry-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="GeoJSON file with search geometry",
)
@click.option(
    "--start-date",
    required=True,
    help="Search start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    required=True,
    help="Search end date (YYYY-MM-DD)",
)
@click.option(
    "--resolution",
    default=20,
    show_default=True,
    help="Output resolution in meters",
)
@click.option(
    "--resampling-method",
    type=click.Choice([m.name for m in ResamplingMethod], case_sensitive=False),
    default="NEAREST",
    show_default=True,
    help="Resampling method",
)
@click.option(
    "--output-folder",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    help="Directory to save downloaded scenes",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing files",
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    help="Parallel download workers",
)
def main(
    catalog_url,
    collection_name,
    raster_assets,
    file_assets,
    max_cloud_cover,
    geometry_path,
    start_date,
    end_date,
    resolution,
    resampling_method,
    output_folder,
    overwrite,
    num_workers,
):
    """
    Download Sentinel-2 Level-2A scenes from a STAC catalog.
    """
    logger = get_logger()
    downloader = STACDownloader(catalog_url=catalog_url, logger=logger)

    logger.info(f"Searching {collection_name} from {start_date} to {end_date}…")
    t0 = time.time()
    geom = gpd.read_file(geometry_path).geometry.values[0]
    items = downloader.query_catalog(
        collection_name=collection_name,
        start_date=start_date,
        end_date=end_date,
        geometry=geom,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    logger.info(f"Found {len(items)} items in {time.time() - t0:.1f}s")

    os.makedirs(output_folder, exist_ok=True)
    method = ResamplingMethod[resampling_method.upper()]
    logger.info(f"Downloading to {output_folder} (overwrite={overwrite})…")
    downloaded = downloader.download_items(
        items=items,
        raster_assets=list(raster_assets),
        file_assets=list(file_assets),
        output_folder=output_folder,
        overwrite=overwrite,
        resolution=resolution,
        resampling_method=method,
        num_workers=num_workers,
    )
    logger.info(f"Downloaded {len(downloaded)} items.")


if __name__ == "__main__":
    main()
