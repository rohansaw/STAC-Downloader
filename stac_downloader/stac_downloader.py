import multiprocessing
import os

import numpy as np
from tqdm import tqdm

from stac_downloader.downloading import download_file, download_raster_file
from stac_downloader.raster_processing import ResamplingMethod, build_bandstacked_vrt, resample_raster, save_band

from pystac.item import Item as pyStacItem

from typing import List, Dict, Tuple
from stac_downloader.utils import get_logger

class STACDownloader:
    def __init__(self, logger=None):

        if logger is None:            
            logger = get_logger()

        self.logger = logger
        self.mask_band_processor = None
        self.bandprocessing_hooks = []
        self.postdownload_hooks = []

    def register_masking_hook(self, hook, masking_assets):
        """
        Register a hook function to process mask bands.
        The hook should take the downloaded mask bands (dict: name-> raster, profile) and return a binary mask.
        """
        if not callable(hook):
            raise ValueError("Hook must be a callable function.")
                
        self.mask_band_processor = hook
        self.mask_assets = masking_assets

    def register_bandprocessing_hook(self, hook, band_assets):
        """
        Register a hook function to adjust a band.
        Args:
            hook (callable): The function to register. Must accept parameters: raster, profile, item.
            band_assets (List[str]): List of asset names that the hook should be applied to.
        """
        self.bandprocessing_hooks.append((hook, band_assets))

    def register_postdownload_hook(self, hook):
        self.postdownload_hooks.append(hook)

    def _get_file_output_path(self, item: pyStacItem, asset_name: str, resolution: float, output_folder: str, extension: str = "tif"):
        f_name = f"{item.id}_{asset_name}_{str(resolution) + 'm' if resolution is not None else ''}.{extension}"
        out_path = os.path.join(output_folder, f_name)
        return out_path

    def _download_file_assets(self, item: pyStacItem, file_assets: List[str], output_folder: str, overwrite: bool):
        file_asset_paths = {}
        if file_assets:
            for file_asset in file_assets:
                if file_asset not in item.assets:
                    raise ValueError(f"Asset '{file_asset}' not found in item.")

                file_url = item.assets[file_asset].href
                file_out_path = self._get_file_output_path(item, file_asset, None, output_folder)
                try:
                    download_file(file_url, file_out_path, overwrite)
                except Exception as e:
                    raise Exception(
                        f"Failed to download file asset '{file_asset}' from URL: {file_url}. Error {e}"
                    )
                file_asset_paths[file_asset] = file_out_path
        
        return file_asset_paths
    
    def _download_raster_assets(self, item: pyStacItem, raster_assets: List[str], output_folder: str, overwrite: bool, resolution: float, resampling_method: ResamplingMethod, mask=None):
        download_paths = {}
        
        for raster_asset in raster_assets:
            if raster_asset not in item.assets:
                raise ValueError(f"Asset '{raster_asset}' not found in item.")

            raster_url = item.assets[raster_asset].href
            raster_out_path = self._get_file_output_path(item, raster_asset, resolution, output_folder)

            # Skip download if the file already exists and not forching overwrite
            if os.path.exists(raster_out_path) and not overwrite:
                self.logger.info(f"Raster asset '{raster_asset}' already exists at {raster_out_path}. Skipping download.")
                download_paths[raster_asset] = raster_out_path
                continue

            try:
                self.logger.info(f"Downloading raster asset '{raster_asset}' from {raster_url}...")
                raster, profile, bounds = download_raster_file(raster_url)
                resampled_raster, resampled_profile = resample_raster(raster, profile, bounds, resolution, resampling_method)

                # Apply mask if provided
                if mask is not None:
                    nodata_value = resampled_profile["nodata"]
                    if nodata_value is None:
                        raise Exception(
                            f"Raster asset '{raster_asset}' does not have a defined nodata value, which is required to apply a mask."
                        )
                    
                    resampled_raster = np.where(mask == 1, resampled_raster, nodata_value)

                raster_out_path = self._save_band(resampled_raster, resampled_profile, item, raster_asset, resolution, output_folder)
            except Exception as e:
                # Cleanup if download failed to avoid corrupted files
                if os.path.exists(raster_out_path):
                    os.remove(raster_out_path)
                raise Exception(
                    f"Failed to download & process raster asset '{raster_asset}' from URL: {raster_url}. Error {e}"
                )
            
            download_paths[raster_asset] = raster_out_path

        return download_paths
    
    def _save_band(self, raster: np.ndarray, profile: dict, item: pyStacItem, asset_name: str, resolution: float, output_folder: str):
        raster_out_path = self._get_file_output_path(item, asset_name, resolution, output_folder)
        try:
            save_band(raster, profile, raster_out_path, asset_name)
        except Exception as e:
            raise Exception(
                f"Failed to save band '{asset_name}' for item '{item.id}' at resolution {resolution}m. Error: {e}"
            )

        return raster_out_path
    
    def _create_mask_from_assets(self, item: pyStacItem, mask_assets: List[str], resolution: float, resampling_method: ResamplingMethod):
        if mask_assets:
            if len(mask_assets) > 1 and not self.mask_band_processor is None:
                raise ValueError("Maskband processing function required for multiple mask bands.")

            self.logger.info("Downloading mask assets and resampling to target resolution...")

            downloaded_maskbands = {}
            for mask_asset in mask_assets:
                if mask_asset not in item.assets:
                    raise ValueError(f"Mask asset '{mask_asset}' not found in item.")

                mask_url = item.assets[mask_asset].href
                self.logger.info(f"Downloading mask asset '{mask_asset}' from {mask_url}...")
                mask, mask_profile, mask_bounds = download_raster_file(mask_url)

                # Resample the mask to the target resolution
                resampled_mask, resampled_mask_profile = resample_raster(
                    mask, mask_profile, mask_bounds, resolution, resampling_method
                )

                downloaded_maskbands[mask_asset] = (resampled_mask_profile, resampled_mask)

            if not self.mask_band_processor and len(mask_assets) == 1:
                mask_metadata, mask = downloaded_maskbands[mask_assets[0]]
            else:
                self.logger.info("Processing mask bands to binary mask.")
                mask_metadata, mask = self.mask_band_processor(downloaded_maskbands, resolution)

            # Val: The mask band should be binary: 1 for pixels to keep and 0 for pixels to mask
            if not np.isin(np.unique(mask), [0, 1]).all():
                raise ValueError(
                    f"Mask must be binary (0 and 1 values only). Values found: {np.unique(mask)}"
                )
            
            return mask, mask_metadata
        
        return None, None
    
    def _run_postdownload_hooks(self, item: pyStacItem, band_paths: Dict[str, str], band_names: List[str], mask: np.ndarray, file_asset_paths: Dict[str, str], resolution: float, output_folder: str) -> Tuple[Dict[str, str], List[str]]:
        for idx, hook in enumerate(self.postdownload_hooks):
            try:
                self.logger.info(f"Running post-download hook {idx}...")
                band_paths, band_names = hook(
                    item, band_paths, band_names, mask, file_asset_paths, resolution, output_folder
                )
            except Exception as e:
                self.logger.error(f"Post-download hook {idx} failed: {e}")
                raise e
        
        return band_paths, band_names

    def _process_item(self, item: pyStacItem, raster_assets: List[str], file_assets: List[str], mask_assets: List[str], output_folder: str, overwrite: bool, resolution: float, resampling_method: ResamplingMethod, save_mask_as_band: bool) -> Tuple[str, Dict[str, str]]:
        # Step 1: Download Metadata / Files that are not processed as rasters
        file_asset_paths = self._download_file_assets(item, file_assets, output_folder, overwrite)

        # Step 2: Download maskbands and build mask. This is done before downloading other bands, to reuse.
        mask, mask_metadata = self._create_mask_from_assets(
            item, mask_assets, resolution, resampling_method
        )

        # Step 3: Download rasters, resample, bandprocess
        self.logger.info("Downloading band data and resampling...")
        band_paths = self._download_raster_assets(item, raster_assets, output_folder, overwrite, resolution, resampling_method, mask=mask)
        band_names_ordered = raster_assets

        # Add mask band if requested
        if save_mask_as_band:
            mask_band_path = self._save_band(mask, mask_metadata, item, "mask", resolution, output_folder)
            band_paths["mask"] = mask_band_path
            band_names_ordered.append("mask")

        # Step 4: Run Postdownload hooks
        self.logger.info("Running post-download hooks...")
        band_paths, band_names_ordered = self._run_postdownload_hooks(item, band_paths, band_names_ordered, mask, file_asset_paths, resolution, output_folder)

        # Step 5: Combine bands into a vrt (or GTiff if requested)
        self.logger.info(f"Combining bands into single file for tile {item.id}...")
        vrt_path = os.path.join(
            output_folder, f"{item.id}_{resolution}m_{item.datetime.strftime('%Y-%m-%d')}.vrt"
        )

        try:
            vrt_path = build_bandstacked_vrt(vrt_path, band_paths, band_names_ordered)
        except Exception as e:
            self.logger.error(f"Failed to build VRT for item {item.id}: {e}")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            raise e

        return vrt_path, file_asset_paths, band_paths, band_names_ordered
    
    def _execution_wrapper(self, args):
        try:
            return self._process_item(*args)
        except Exception as e:
            item_id = args[0].id if args and args[0] and hasattr(args[0], "id") else "unknown"
            raise type(e)(f"Error processing item {item_id}: {str(e)}") from e
        
    def _check_for_existing_output(self, items: List[pyStacItem], output_folder: str, resolution: float, output_paths: List[str], overwrite: bool):
        filtered_items = []
        for item in items:
            output_path = os.path.join(
                output_folder, f"{item.id}_{resolution}m_{item.datetime.strftime('%Y-%m-%d')}.vrt"
            )
            if not os.path.exists(output_path) or overwrite:
                filtered_items.append(item)
        
        return filtered_items

    def download_items(self, items: List[pyStacItem], raster_assets: List[str], file_assets: List[str], mask_assets: List[str], output_folder: str, overwrite: bool, resolution: float, resampling_method: ResamplingMethod, save_mask_as_band: bool = False, num_workers: int = 1):
        # Filter items by checkig it output already exists
        if os.path.exists(output_folder):
            if not overwrite:
                items = self._check_for_existing_output(items, output_folder, resolution, output_paths, overwrite)
        else:
            os.makedirs(output_folder)

        job_args = [
            (
                item,
                raster_assets,
                file_assets,
                mask_assets,
                output_folder,
                overwrite,
                resolution,
                resampling_method,
                save_mask_as_band,
            )
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
                self.logger.error("\nInterrupted by user. Terminating workers...")
                pool.terminate()
                raise
            except Exception as e:
                self.logger.error(f"\nError in worker process: {e}")
                pool.terminate()
                raise

        return outputs
