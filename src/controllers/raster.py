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
