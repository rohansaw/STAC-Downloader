from collections import defaultdict
import multiprocessing
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from pystac.item import Item as pyStacItem
from pystac_client import Client as pyStacClient
from rasterio import Env
from tenacity import retry, stop_after_attempt, wait_exponential
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
from stac_downloader.utils import get_logger, run_subprocess


class STACDownloader:
    def __init__(self, catalog_url=None, logger=None, stac_item_modifier=None):

        if logger is None:
            logger = get_logger()

        self.stac_item_modifier = stac_item_modifier 

        self.logger = logger
        self.catalog = pyStacClient.open(catalog_url) if catalog_url else None
        self.masking_hook = None
        self.bandprocessing_hooks = defaultdict(list)
        self.postdownload_hooks = []

        self._check_requirements()

    def _check_requirements(self):
        try:
            run_subprocess(["gdalinfo", "--version"], "Check GDAL installation", self.logger)
        except RuntimeError:
            raise RuntimeError("GDAL is not installed or not in the PATH.")

    def register_masking_hook(self, hook):
        """
        Register a hook function to create a mask from the mask ban.
        
        The hook should take the downloaded mask bands (dict: name-> profile, mask)
        and return a binary mask as: profile, binary_mask.
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
        if not callable(hook):
            raise ValueError("Hook must be a callable function.")
        
        for asset_name in band_assets:
            self.bandprocessing_hooks[asset_name].append(hook)


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
        modifier=None,
        **kwargs,
    ) -> List[pyStacItem]:
        modifier = modifier if modifier else self.stac_item_modifier
        if catalog_url:
            catalog = pyStacClient.open(catalog_url, modifier=modifier)
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

    def _download_file_assets(
        self, item: pyStacItem, file_assets: List[str], output_folder: str
    ):
        file_asset_paths = {}
        if file_assets:
            for file_asset in file_assets:
                if file_asset not in item.assets:
                    raise ValueError(f"Asset '{file_asset}' not found in item.")

                file_url = item.assets[file_asset].href
                ext = os.path.splitext(file_url)[-1].lstrip(".")
                ext = ext.split('?')[0] # Removing signing key
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
    
    def _process_band(self, raster: np.ndarray, profile: dict, item: pyStacItem, asset_name: str):
        processors = self.bandprocessing_hooks[asset_name]

        if not processors:
            return raster, profile
        
        for processor in processors:
            raster, profile = processor(raster, profile, item)

        return raster, profile

    def _download_raster_assets(
        self,
        item: pyStacItem,
        raster_assets: List[str],
        output_folder: str,
        resolution: float,
        resampling_method: ResamplingMethod,
        mask=None,
        raster_asset_target_dtypes: Dict[str: Any] = None
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

                    processed_raster, processed_profile = self._process_band(resampled_raster, resampled_profile, item, raster_asset)

                    # Apply mask if provided
                    if mask is not None:
                        nodata_value = processed_profile["nodata"]
                        if nodata_value is None:
                            self.logger.warning(
                                f"Raster asset '{raster_asset}' does not have a defined nodata value. Using 0."
                            )
                            nodata_value = 0

                        processed_raster = apply_mask(
                            processed_raster, mask, nodata_value
                        )

                    raster_out_path = self._save_band(
                        processed_raster,
                        processed_profile,
                        item,
                        raster_asset,
                        resolution,
                        output_folder,
                        raster_asset_target_dtypes
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
        raster_asset_target_dtypes: Dict[str, Any]
    ):
        raster_out_path = self._get_file_output_path(
            item, asset_name, resolution, output_folder
        )
        try:

            if not raster_asset_target_dtypes or not asset_name in raster_asset_target_dtypes:
                self.logger.warning("No dtype specified. Falling back to int16")
                dtype = np.int16
            else:
                dtype = raster_asset_target_dtypes[asset_name]
            save_band(raster, profile, raster_out_path, asset_name, dtype)
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
                raise ValueError(
                    "Maskband processing function required for multiple mask bands."
                )

            self.logger.info(
                "Downloading mask assets and resampling to target resolution..."
            )

            downloaded_maskbands = {}
            for mask_asset in mask_assets:
                if mask_asset not in item.assets:
                    raise ValueError(f"Mask asset '{mask_asset}' not found in item.")

                mask_url = item.assets[mask_asset].href
                self.logger.info(
                    f"Downloading mask asset '{mask_asset}' from {mask_url}..."
                )

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

    def _get_vrt_output_path(
        self, item: pyStacItem, resolution: float, output_folder: str
    ) -> str:
        return os.path.join(
            output_folder,
            f"{item.id}_{resolution}m_{item.datetime.strftime('%Y-%m-%d')}.vrt",
        )
    
    def _modify_item(self, item):
        if self.stac_item_modifier:
            item =  self.stac_item_modifier(item)
        
        return item
        
    # Using retry since we are sometimes reading from remote filesystems (e.g. S3) and they can be flaky.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
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
        raster_asset_target_dtypes: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str]]:
        
        # Step 0: Modify an item. Typically used to sign an item
        item = self._modify_item(item)

        # Step 1: Download Metadata / Files that are not processed as rasters
        file_asset_paths = self._download_file_assets(item, file_assets, output_folder)

        # Step 2: Download maskbands and build mask. This is done before downloading other bands, to reuse.
        mask, mask_metadata = self._create_mask_from_assets(
            item, mask_assets, resolution, resampling_method,
        )

        # Step 3: Download rasters, resample, bandprocess
        self.logger.info("Downloading band data and resampling...")
        band_paths = self._download_raster_assets(
            item, raster_assets, output_folder, resolution, resampling_method, mask=mask, raster_asset_target_dtypes=raster_asset_target_dtypes
        )
        band_names_ordered = raster_assets

        # Add mask band if requested
        if save_mask_as_band:
            mask_band_path = self._save_band(
                mask, mask_metadata, item, "mask", resolution, output_folder, raster_asset_target_dtypes
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

    def _execution_wrapper(self, kwargs_dict):
        try:
            return self.download_item(**kwargs_dict)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Download interrupted by user.")
        except Exception as e:
            item_id = (
                kwargs_dict.get("item", {}).id
                if kwargs_dict.get("item") and hasattr(kwargs_dict.get("item"), "id")
                else "unknown"
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
        n_already_downloaded = 0
        for item in items:
            output_path = self._get_vrt_output_path(item, resolution, output_folder)
            if not os.path.exists(output_path) or overwrite:
                filtered_items.append(item)
            else:
                n_already_downloaded += 1

        self.logger.info(f'{n_already_downloaded}/{len(items)} items already downloaded.')

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
        raster_asset_target_dtypes: Dict[str, Any] = None
    ):
        # Filter items by checkig it output already exists
        if os.path.exists(output_folder):
            if not overwrite:
                items = self._check_for_existing_output(
                    items, output_folder, resolution, overwrite
                )
        else:
            os.makedirs(output_folder)

        if len(items) == 0:
            self.logger.info('All items already downloaded. Exiting.')
            return

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
                "raster_asset_target_dtypes": raster_asset_target_dtypes
            }
            for item in items
        ]

        self.logger.info(f"Using {num_workers} workers out of {multiprocessing.cpu_count()} available cores")

        outputs = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            try:
                results_iter = pool.imap_unordered(self._execution_wrapper, job_args)

                with tqdm(total=len(job_args), desc="Processing items") as pbar:
                    for result in results_iter:
                        outputs.append(result)
                        pbar.update(1)

            except KeyboardInterrupt:
                self.logger.error("\nInterrupted by user. Terminating workers...")
                pool.terminate()
                pool.join()
                raise KeyboardInterrupt("Download interrupted by user.")
            except Exception as e:
                self.logger.error(f"\nError in worker process: {e}")
                pool.close()
                pool.join()
                raise e

        return outputs
