"""
This module contains the RasterController class.
"""
import rasterio as rio


class RasterController:
    """
    This class represents a RasterController.
    """

    def open_raster(self, raster):
        """
        Opens a raster file using rasterio.

        Args:
            raster (str): Path to the raster file.

        Returns:
            rasterio.DatasetReader: The opened raster dataset.

        Opens a raster file using rasterio library and returns the opened raster dataset.

        Args:
            raster (str): Path to the raster file.

        Returns:
            rasterio.DatasetReader: The opened raster dataset.
        """
        dataset = rio.open(raster, dtype="uint8")
        num_bands = dataset.count
        print(f"Number of bands in the image: {num_bands}\n")

        rows, cols = dataset.shape
        print(f"Image size: {rows} rows x {cols} columns\n")

        driver = dataset.driver
        print(f"Raster driver: {driver}\n")

        proj = dataset.crs
        print(f"Image projection:\n {proj}\n")

        return dataset

    def read_raster(self, raster, window=None):
        """
        Reads data from a raster file.

        Args:
            raster (rasterio.DatasetReader): The raster dataset.
            window: A window to specify the subset of the raster to read. Defaults to None.

        Returns:
            numpy.ndarray: The data read from the raster.

        """
        if window:
            return raster.read(window=window)
        return raster.read()

    def merge_rasters(self, band_paths, product_id):
        """
        Merge multiple images into a single composite image.

        Parameters:
            band_paths (List[str]): A list of file paths to the individual band images.
            product_id (str): The ID of the product.

        Returns:
            str: The file path of the merged image.
        """
        #name = band_paths[0].split("/")[-1].split("_B")[0]
        # Read metadata of the first file and assume all other bands are the same
        with rio.open(band_paths[0]) as src0:
            meta = src0.meta

        # Update metadata to reflect the number of layers and set LWZ compression
        meta.update(count=len(band_paths), compress='LZW')
        
        # Create the combined image and write the bands to it
        output_dir = f'../data/images/processed/products/landsat/{product_id}_M.tif'
        with rio.open(output_dir, 'w', **meta) as dst:
            for band_id, band_path in enumerate(band_paths, start=1):
                with rio.open(band_path) as src1:
                    dst.write_band(band_id, src1.read(1))
                    
        return output_dir