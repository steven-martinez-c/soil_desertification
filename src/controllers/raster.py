"""
This module contains the RasterController class.
"""
import numpy as np
import rasterio as rio
from rasterio import mask
from rasterio.windows import Window
import geopandas as gpd
from src.layers.utilities import search_mtl_params, cloud_masking


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
    
    
    def read_windows(self, rasters, c, r, buffer, tile_size):
        """
        Reads windows from multiple rasters.

        Args:
            rasters (list): A list of rasters.
            c (int): The column index.
            r (int): The row index.
            buffer (int): The buffer size.
            tile_size (int): The size of the tile.

        Returns:
            tuple: A tuple containing the tiles.
        """
        tiles = []
        #only works when rasters are in same projection
        for raster in rasters:
            tile = raster.read(list(np.arange(1, raster.count+1)), window=Window(c-buffer, r-buffer, tile_size, tile_size))
            tiles.append(tile)
        return (*tiles,)

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

    def cut_shape_mask(self, dataset, mask_shape):
        """
        Crop the input raster image using the provided shapefile mask.

        Args:
            dataset (str): Path to the input raster image.
            mask_shape (str): Path to the shapefile mask.
            product_id (str): Product ID.

        Returns:
            str: Path to the cropped raster image.
        """
        try:
            # Open the input raster image
            with rio.open(dataset) as src:
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

                product_id = dataset.split("/")[-1].split(".tif")[0]
                # Define the output path for the cropped image
                output_path = (
                    f"../data/images/processed/products/landsat/clips/{product_id}C.tif"
                )

                # Write the cropped image to the output path
                with rio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

            return output_path

        except Exception as error:
            print(f"An error occurred while cropping the image: {str(error)}")

    def correct_toa_radiance(self, dataset_raster, metadata):
        """
        Corrects top-of-atmosphere (TOA) radiance values in an image using provided metadata.

        Args:
            imagen (str): Path to the Landsat image file.
            metadata (str): Metadata containing information required for correction.

        Returns:
            str: Path to the corrected TOA image file.

        Raises:
            Exception: If an error occurs during the correction process.

        """
        try:
            # Extract solar elevation angle from metadata
            solar_elevation = float(search_mtl_params("SUN_ELEVATION", metadata))
            solar_angle = np.sin(
                np.radians(solar_elevation)
            )  # angulo solar zenith local

            # Read the input image
            with rio.open(dataset_raster) as src:
                data = src.read()  # Read all bands

            # Create metadata for the output corrected image
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff", "compress": "LZW", "dtype": "uint16"})

            # Create an array to store corrected data
            corrected_data = np.empty_like(data, dtype=np.uint16)

            for index, q_cal in enumerate(data):
                if index < 7:
                    # Extract multiplicative and additive reflectance scaling factors
                    mul_ref = float(
                        search_mtl_params(
                            f"REFLECTANCE_MULT_BAND_{index + 1}", metadata
                        )
                    )
                    add_ref = float(
                        search_mtl_params(f"REFLECTANCE_ADD_BAND_{index + 1}", metadata)
                    )

                    # Calculate TOA reflectance
                    planetary_reflectance = (mul_ref * q_cal + add_ref) / solar_angle

                    # Set nodata values to NaN
                    planetary_reflectance[q_cal == out_meta["nodata"]] = 0

                    # Ensure reflectance values are within the 0-1 range
                    planetary_reflectance = np.clip(planetary_reflectance, 0, 1)

                    # Scaling the values to the appropriate range for uint16 (0-65535)
                    scaled_planetary_reflectance = (
                        planetary_reflectance * 65535
                    ).astype(np.uint16)

                    corrected_data[index] = scaled_planetary_reflectance
                else:
                    corrected_data[index] = q_cal

            # Extract product ID from metadata
            product_id = dataset_raster.split("/")[-1].split(".tif")[0]

            # Define the output file path for the corrected image
            output_file = (
                f"../data/images/processed/products/landsat/toa/{product_id}T.tif"
            )

            # Write the corrected data to the output file
            with rio.open(output_file, "w", **out_meta) as dest:
                dest.write(corrected_data)

            return output_file

        except Exception as error:
            print(f"Error en la corrección TOA: {str(error)}")


    def cut_cloud_mask(self, image):
        """
        Cut cloud mask from dataset.

        Args:
            dataset (ndarray): The input dataset.

        Returns:
            ndarray: The masked dataset with cloud pixels set to 1.
            meta: The metadata of the masked dataset.
        """
        with rio.open(image) as src:
            dataset = src.read()
        
            # Get the QA band
            qa_band = dataset[7, :, :]
            # Set any value of -9999 to 0
            qa_band[qa_band == -9999] = 0

            # Generate the cloud mask using the cloud_masking function
            cloud_mask = [0 if cloud_masking(value) else 1 for value in qa_band.flatten()]
            
            # Reshape the cloud mask to match the shape of the dataset
            cloud_bits = np.array(cloud_mask).reshape(dataset[0].shape)
            
            # Update the metadata of the output image
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": dataset.shape[1],
                    "width": dataset.shape[2],
                    "compress": "LZW",
                }
            )

            # Create a copy of the dataset
            masked_data = np.copy(dataset)

            # Set the cloud pixels in the masked_data to 0
            masked_data[:-1, cloud_bits == 0] = 0
            
            product_id = image.split("/")[-1].split(".tif")[0]
            output = f'../data/images/processed/products/landsat/cloud/{product_id}.tif'
            with rio.open(output, "w", **out_meta) as dest:
                dest.write(masked_data)

        return masked_data, cloud_mask, src.meta


    def apply_mask_without_clouds(self, image, clouds, cloud_mask, meta, product_id):
        """
        Apply a cloud mask to an image, removing the clouds.

        Args:
            image (str): Path to the image file.
            clouds (ndarray): Array containing the cloud data.
            cloud_mask (ndarray): Array containing the cloud mask.
            cloud_bits (ndarray): Array containing the cloud bits.
            meta (dict): Metadata for the output file.

        Returns:
            ndarray: The image with clouds masked out.
        """
        # Open the image file
        with rio.open(image) as src:
            # Read the image data
            data_without_clouds = src.read()
            
        # Reshape the cloud mask to match the shape of the dataset
        cloud_bits = np.array(cloud_mask).reshape(clouds[0].shape)

        # Crop data_without_clouds to match the dimension of cloud_bits
        data_without_clouds = data_without_clouds[:, :, : cloud_bits.shape[1]]
        
        # Create a copy of data_without_clouds
        masked_data = np.copy(data_without_clouds)
        
        # Check if clouds has more bands than data_without_clouds
        if clouds.shape[2] > masked_data.shape[2]:
            diff = clouds.shape[2] - masked_data.shape[2]
            masked_data = np.pad(masked_data, ((0, 0), (0, 0), (0, diff)), mode='constant', constant_values=0)
        else:
            masked_data = masked_data

        # Invert the cloud mask
        cloud_mask_inverse = cloud_bits != 1

        # Replace the cloud pixels in data_clouds with the corresponding pixels in masked_data
        clouds[:-1, cloud_mask_inverse] = masked_data[:-1, cloud_mask_inverse]

        product = product_id.split(".tif")[0]
        output = f'../data/images/processed/products/landsat/toa/{product}.tif'
        with rio.open(output, "w", **meta) as dest:
            dest.write(clouds)

        return clouds