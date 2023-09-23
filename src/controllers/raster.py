"""
This module contains the RasterController class.
"""
import rasterio as rio
from rasterio import mask
import geopandas as gpd


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
        # name = band_paths[0].split("/")[-1].split("_B")[0]
        # Read metadata of the first file and assume all other bands are the same
        with rio.open(band_paths[0]) as src0:
            meta = src0.meta

        # Update metadata to reflect the number of layers and set LWZ compression
        meta.update(count=len(band_paths), compress="LZW")

        # Create the combined image and write the bands to it
        output_dir = f"../data/images/processed/products/landsat/{product_id}_M.tif"
        with rio.open(output_dir, "w", **meta) as dst:
            for band_id, band_path in enumerate(band_paths, start=1):
                with rio.open(band_path) as src1:
                    dst.write_band(band_id, src1.read(1))

        return output_dir

    def crop_mask_raster(self, image, mask_shape, product_id):
        """
        Crop the input raster image using the provided shapefile mask.

        Args:
            image_path (str): Path to the input raster image.
            mask_shape_path (str): Path to the shapefile mask.
            product_id (str): Product ID.

        Returns:
            str: Path to the cropped raster image.
        """
        try:
            # Open the input raster image
            with rio.open(image) as src:
                # Read and reproject the shapefile mask
                shapefile = gpd.read_file(mask_shape)
                shapefile = shapefile.to_crs(src.crs)

                # Crop the image using the shapefile mask
                out_image, out_transform = mask.mask(src, shapefile.geometry, crop=True)

                # Update the metadata of the output image
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "LZW",
                    }
                )

                # Define the output path for the cropped image
                output_path = (
                    f"../data/images/processed/products/landsat/crops/{product_id}C.tif"
                )

                # Write the cropped image to the output path
                with rio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

            return output_path
        
        except Exception as error:
            print(f"An error occurred while cropping the image: {str(error)}")
