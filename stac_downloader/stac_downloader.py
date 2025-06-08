import multiprocessing
import os
from typing import Dict, List, Tuple

import numpy as np
from pystac.item import Item as pyStacItem
from pystac_client import Client as pyStacClient
from rasterio import Env
from tqdm import tqdm

from stac_downloader.downloading import download_file
from stac_downloader.raster_processing import (
    ResamplingMethod,
    apply_mask,
    build_bandstacked_vrt,
    is_binary,
    resample_raster,
    save_band,
)
from stac_downloader.utils import get_logger


class STACDownloader:
    def __init__(self, catalog_url=None, logger=None):

        if logger is None:
            logger = get_logger()

        self.logger = logger
        self.catalog = pyStacClient.open(catalog_url) if catalog_url else None
        self.masking_hook = None
        self.bandprocessing_hooks = []
        self.postdownload_hooks = []

    def register_masking_hook(self, hook):
        """
        Register a hook function to create a mask from the mask ban.
        The hook should take the downloaded mask bands (dict: name-> profile, mask) and return a binary mask as: profile, binary_mask.
        """
        if not callable(hook):
            raise ValueError("Hook must be a callable function.")

        self.masking_hook = hook

    def register_bandprocessing_hook(self, hook, band_assets: List[str]):
        """
        Register a hook function to adjust a band.
        Args:
            hook (callable): The function to register. Must accept parameters: raster, profile, item.
            band_assets (List[str]): List of asset names that the hook should be applied to.
        """
        raise NotImplementedError(
            "Band processing hooks are not implemented yet. Use postdownload hooks instead."
        )

    def register_postdownload_hook(self, hook):
        if not callable(hook):
            raise ValueError("Hook must be a callable function.")
        self.postdownload_hooks.append(hook)

    def query_catalog(
        self,
        collection_name: str,
        start_date: str,
        end_date: str,
        geometry=None,
        bbox=None,
        query=None,
        catalog_url=None,
        **kwargs,
    ) -> List[pyStacItem]:
        if catalog_url:
            catalog = pyStacClient.open(catalog_url)
        else:
            if self.catalog is None:
                raise ValueError("Catalog URL must be provided to query the catalog.")
            catalog = self.catalog

        collection = catalog.get_collection(collection_name)

        if not collection:
            raise ValueError(f"Collection '{collection_name}' not found in catalog.")

        search_params = {
            "collections": [collection_name],
            "datetime": f"{start_date}/{end_date}",
            "query": query,
            **kwargs,
        }

        if geometry is not None:
            search_params["intersects"] = geometry
        if bbox is not None:
            search_params["bbox"] = bbox

        search = catalog.search(**search_params)
        items = search.item_collection()
        return items

    def _get_file_output_path(
        self,
        item: pyStacItem,
        asset_name: str,
        resolution: float,
        output_folder: str,
        extension: str = "tif",
    ):
        f_name = f"{item.id}_{asset_name}{'_' + str(resolution) + 'm' if resolution is not None else ''}.{extension}"
        out_path = os.path.join(output_folder, f_name)
        return out_path

    def _download_file_assets(self, item: pyStacItem, file_assets: List[str], output_folder: str):
        file_asset_paths = {}
        if file_assets:
            for file_asset in file_assets:
                if file_asset not in item.assets:
                    raise ValueError(f"Asset '{file_asset}' not found in item.")

                file_url = item.assets[file_asset].href
                ext = os.path.splitext(file_url)[-1].lstrip(".")
                file_out_path = self._get_file_output_path(
                    item, file_asset, None, output_folder, extension=ext
                )
                try:
                    download_file(file_url, file_out_path, overwrite=True)
                except Exception as e:
                    raise Exception(
                        f"Failed to download file asset '{file_asset}' from URL: {file_url}. Error {e}"
                    )
                file_asset_paths[file_asset] = file_out_path

        return file_asset_paths

    def _download_raster_assets(
        self,
        item: pyStacItem,
        raster_assets: List[str],
        output_folder: str,
        resolution: float,
        resampling_method: ResamplingMethod,
        mask=None,
    ):
        download_paths = {}

        with Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_ENABLE_CURL_MULTI="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            GDAL_HTTP_MULTIPLEX="YES",
            GDAL_HTTP_VERSION="2",
        ):

            for raster_asset in raster_assets:
                if raster_asset not in item.assets:
                    raise ValueError(f"Asset '{raster_asset}' not found in item.")

                raster_url = item.assets[raster_asset].href
                raster_out_path = self._get_file_output_path(
                    item, raster_asset, resolution, output_folder
                )

                try:
                    self.logger.info(
                        f"Fetching raster asset '{raster_asset}' from {raster_url} and resampling ..."
                    )
                    resampled_raster, resampled_profile = resample_raster(
                        raster_url, resolution, resampling_method
                    )

                    # Apply mask if provided
                    if mask is not None:
                        nodata_value = resampled_profile["nodata"]
                        if nodata_value is None:
                            raise Exception(
                                f"Raster asset '{raster_asset}' does not have a defined nodata value, which is required to apply a mask."
                            )

                        resampled_raster = apply_mask(resampled_raster, mask, nodata_value)

                    raster_out_path = self._save_band(
                        resampled_raster,
                        resampled_profile,
                        item,
                        raster_asset,
                        resolution,
                        output_folder,
                    )
                except Exception as e:
                    # Cleanup if download failed to avoid corrupted files
                    if os.path.exists(raster_out_path):
                        os.remove(raster_out_path)
                    raise Exception(
                        f"Failed to download & process raster asset '{raster_asset}' from URL: {raster_url}. Error {e}"
                    )

                download_paths[raster_asset] = raster_out_path

            return download_paths

    def _save_band(
        self,
        raster: np.ndarray,
        profile: dict,
        item: pyStacItem,
        asset_name: str,
        resolution: float,
        output_folder: str,
    ):
        raster_out_path = self._get_file_output_path(item, asset_name, resolution, output_folder)
        try:
            save_band(raster, profile, raster_out_path, asset_name)
        except Exception as e:
            raise Exception(
                f"Failed to save band '{asset_name}' for item '{item.id}' at resolution {resolution}m. Error: {e}"
            )

        return raster_out_path

    def _create_mask_from_assets(
        self,
        item: pyStacItem,
        mask_assets: List[str],
        resolution: float,
        resampling_method: ResamplingMethod,
    ):
        if mask_assets:
            if len(mask_assets) > 1 and self.masking_hook is None:
                raise ValueError("Maskband processing function required for multiple mask bands.")

            self.logger.info("Downloading mask assets and resampling to target resolution...")

            downloaded_maskbands = {}
            for mask_asset in mask_assets:
                if mask_asset not in item.assets:
                    raise ValueError(f"Mask asset '{mask_asset}' not found in item.")

                mask_url = item.assets[mask_asset].href
                self.logger.info(f"Downloading mask asset '{mask_asset}' from {mask_url}...")

                # Resample the mask to the target resolution
                resampled_mask, resampled_mask_profile = resample_raster(
                    mask_url, resolution, resampling_method
                )

                downloaded_maskbands[mask_asset] = (
                    resampled_mask_profile,
                    resampled_mask,
                )

            if not self.masking_hook and len(mask_assets) == 1:
                mask_metadata, mask = downloaded_maskbands[mask_assets[0]]
            else:
                self.logger.info("Processing mask bands to binary mask.")
                mask_metadata, mask = self.masking_hook(downloaded_maskbands)

            # Val: The mask band should be binary: 1 for pixels to keep and 0 for pixels to mask
            if not is_binary(mask):
                raise ValueError(
                    f"Mask must be binary (0 and 1 values only). Values found: {np.unique(mask)}"
                )

            return mask, mask_metadata

        return None, None

    def _run_postdownload_hooks(
        self,
        item: pyStacItem,
        band_paths: Dict[str, str],
        band_names: List[str],
        mask: np.ndarray,
        file_asset_paths: Dict[str, str],
        resolution: float,
        output_folder: str,
    ) -> Tuple[Dict[str, str], List[str]]:
        for idx, hook in enumerate(self.postdownload_hooks):
            try:
                self.logger.info(f"Running post-download hook {idx}...")
                band_paths, band_names = hook(
                    item,
                    band_paths,
                    band_names,
                    mask,
                    file_asset_paths,
                    resolution,
                    output_folder,
                )
            except Exception as e:
                self.logger.error(f"Post-download hook {idx} failed: {e}")
                raise e

        return band_paths, band_names

    def _get_vrt_output_path(self, item: pyStacItem, resolution: float, output_folder: str) -> str:
        return os.path.join(
            output_folder,
            f"{item.id}_{resolution}m_{item.datetime.strftime('%Y-%m-%d')}.vrt",
        )

    def download_item(
        self,
        item: pyStacItem,
        raster_assets: List[str],
        file_assets: List[str],
        mask_assets: List[str],
        output_folder: str,
        resolution: float,
        resampling_method: ResamplingMethod,
        save_mask_as_band: bool,
    ) -> Tuple[str, Dict[str, str]]:
        # Step 1: Download Metadata / Files that are not processed as rasters
        file_asset_paths = self._download_file_assets(item, file_assets, output_folder)

        # Step 2: Download maskbands and build mask. This is done before downloading other bands, to reuse.
        mask, mask_metadata = self._create_mask_from_assets(
            item, mask_assets, resolution, resampling_method
        )

        # Step 3: Download rasters, resample, bandprocess
        self.logger.info("Downloading band data and resampling...")
        band_paths = self._download_raster_assets(
            item, raster_assets, output_folder, resolution, resampling_method, mask=mask
        )
        band_names_ordered = raster_assets

        # Add mask band if requested
        if save_mask_as_band:
            mask_band_path = self._save_band(
                mask, mask_metadata, item, "mask", resolution, output_folder
            )
            band_paths["mask"] = mask_band_path
            band_names_ordered.append("mask")

        # Step 4: Run Postdownload hooks
        self.logger.info("Running post-download hooks...")
        band_paths, band_names_ordered = self._run_postdownload_hooks(
            item,
            band_paths,
            band_names_ordered,
            mask,
            file_asset_paths,
            resolution,
            output_folder,
        )

        # Step 5: Combine bands into a vrt (or GTiff if requested)
        self.logger.info(f"Combining bands into single file for tile {item.id}...")
        vrt_path = self._get_vrt_output_path(item, resolution, output_folder)

        try:
            vrt_path = build_bandstacked_vrt(vrt_path, band_paths, band_names_ordered)
        except Exception as e:
            self.logger.error(f"Failed to build VRT for item {item.id}: {e}")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            raise e

        return vrt_path, file_asset_paths, band_paths, band_names_ordered

    def _cleanup_failed_job_artifacts(
        self,
        item: pyStacItem,
        raster_assets: List[str],
        file_assets: List[str],
        output_folder: str,
        resolution: float,
        save_mask_as_band: bool,
    ):
        possible_outputs = []
        for raster_asset in raster_assets:
            possible_outputs.append(
                self._get_file_output_path(item, raster_asset, resolution, output_folder)
            )

        for file_asset in file_assets:
            ext = os.path.splitext(item.assets[file_asset].href)[-1].lstrip(".")
            possible_outputs.append(
                self._get_file_output_path(item, file_asset, None, output_folder, extension=ext)
            )

        if save_mask_as_band:
            possible_outputs.append(
                self._get_file_output_path(item, "mask", resolution, output_folder)
            )

        possible_outputs.append()

        for output_path in possible_outputs:
            if os.path.exists(output_path):
                self.logger.warning(f"Cleaning up failed job artifact: {output_path}")
                os.remove(output_path)

    def _execution_wrapper(self, kwargs_dict):
        try:
            return self.download_item(**kwargs_dict)
        except KeyboardInterrupt:
            self._cleanup_failed_job_artifacts(
                item=kwargs_dict["item"],
                raster_assets=kwargs_dict["raster_assets"],
                file_assets=kwargs_dict["file_assets"],
                output_folder=kwargs_dict["output_folder"],
                resolution=kwargs_dict["resolution"],
                save_mask_as_band=kwargs_dict["save_mask_as_band"],
            )
            raise KeyboardInterrupt("Download interrupted by user.")
        except Exception as e:
            item_id = (
                kwargs_dict.get("item", {}).id
                if kwargs_dict.get("item") and hasattr(kwargs_dict.get("item"), "id")
                else "unknown"
            )
            self._cleanup_failed_job_artifacts(
                item=kwargs_dict["item"],
                raster_assets=kwargs_dict["raster_assets"],
                file_assets=kwargs_dict["file_assets"],
                output_folder=kwargs_dict["output_folder"],
                resolution=kwargs_dict["resolution"],
                save_mask_as_band=kwargs_dict["save_mask_as_band"],
            )
            raise type(e)(f"Error processing item {item_id}: {str(e)}") from e

    def _check_for_existing_output(
        self,
        items: List[pyStacItem],
        output_folder: str,
        resolution: float,
        overwrite: bool,
    ):
        filtered_items = []
        for item in items:
            output_path = self._get_vrt_output_path(item, resolution, output_folder)
            if not os.path.exists(output_path) or overwrite:
                filtered_items.append(item)

        return filtered_items

    def download_items(
        self,
        items: List[pyStacItem],
        raster_assets: List[str],
        file_assets: List[str],
        mask_assets: List[str],
        output_folder: str,
        overwrite: bool,
        resolution: float,
        resampling_method: ResamplingMethod,
        save_mask_as_band: bool = False,
        num_workers: int = 1,
    ):
        # Filter items by checkig it output already exists
        if os.path.exists(output_folder):
            if not overwrite:
                items = self._check_for_existing_output(items, output_folder, resolution, overwrite)
        else:
            os.makedirs(output_folder)

        job_args = [
            {
                "item": item,
                "raster_assets": raster_assets,
                "file_assets": file_assets,
                "mask_assets": mask_assets,
                "output_folder": output_folder,
                "resolution": resolution,
                "resampling_method": resampling_method,
                "save_mask_as_band": save_mask_as_band,
            }
            for item in items
        ]

        outputs = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            try:
                results_iter = pool.imap_unordered(self._execution_wrapper, job_args)

                with tqdm(total=len(job_args), desc="Processing items") as pbar:
                    for result in results_iter:
                        outputs.append(result)
                        pbar.update(1)

            except KeyboardInterrupt:
                # TODO ensure that if we terminate we clean incomplete output
                self.logger.error("\nInterrupted by user. Terminating workers...")
                pool.close()
                pool.join()
                raise KeyboardInterrupt("Download interrupted by user.")
            except Exception as e:
                self.logger.error(f"\nError in worker process: {e}")
                pool.close()
                pool.join()
                raise e

        return outputs
