"""
    This class is responsible for generating tiles from image datasets and label datasets.
"""
import math
import numpy as np
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from pyproj import Proj, Transformer
from src.layers.dict_class import LandCoverClassDict
from src.controllers.raster import RasterController

class TileController:
    def __init__(self, images, label, height, width, pixel_locations, batch_size):
        """
        Initializes the object with the given parameters.

        Args:
            images (list): A list of image datasets.
            label (Dataset): The label dataset.
            height (int): The height of each tile.
            width (int): The width of each tile.
            pixel_locations (list): A list of pixel locations.
            batch_size (int): The batch size.
        """
        
        self.image_datasets = images
        self.label_dataset = label
        self.tile_height = height
        self.tile_width = width
        self.pixel_locations = pixel_locations
        self.batch_size = batch_size

        self.raster_proj = Proj(self.image_datasets['2018'][0].crs)
        self.label_proj = Proj(self.label_dataset.crs)
        self.band_count = self.image_datasets['2018'][0].count
        self.class_count = len(LandCoverClassDict().get_landsat_dictionary())
        self.buffer = math.ceil(self.tile_height / 2)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Generates batches of image and label data for training.

        Yields:
            tuple: A tuple containing the image batch and label batch.
        """
        # Initialize variables
        col = row = 0
        index = 0
        while True:
            # Create empty arrays for image batch and label batch
            image_batch = np.zeros(
                (
                    self.batch_size,
                    self.tile_height,
                    self.tile_width,
                    self.band_count - 1,
                )
            )
            label_batch = np.zeros((self.batch_size, self.class_count))
            count = 0
            while count < self.batch_size:
                # Check if we reached the end of the pixel locations, reset index if true
                if index >= len(self.pixel_locations):
                    index = 0
                # Get row, col, and dataset index from pixel locations
                row, col = self.pixel_locations[index][0]
                dataset_index = self.pixel_locations[index][1]
                index += 1
                # Read tile using image_datasets and specified bands and window
                tile_to_read = RasterController().read_windows(
                    self.image_datasets[dataset_index], col, row, self.buffer, self.tile_height
                )
                
                if self.is_valid_tile(tile_to_read):
                    for item in tile_to_read:
                        # Remove QA band
                        tile = item[0:7]
                        # Reshape tile and standardize
                        reshaped_tile = (reshape_as_image(tile) - 982.5) / 1076.5
                        # Get label and one-hot encode
                        label = self.get_label(dataset_index, row, col)
                        if label != 0 and not np.isnan(label):
                            label_batch[0][label] = 1
                            image_batch[0] = reshaped_tile
                            count += 1
            yield (image_batch, label_batch)

    def is_valid_tile(self, tile):
        """
        Check if a tile is valid based on certain conditions.

        Args:
            tile (numpy.ndarray): The tile to be checked.

        Returns:
            bool: True if the tile is valid, False otherwise.
        """
        for item in tile:
            #print(item.shape)
            # Check if the tile has a size of 0
            if item.size == 0:
                return False

            # Check if the maximum value in the tile is 0
            if np.amax(item) == 0:
                return False

            # Check for specific values in the tile
            if np.isnan(item).any() or -9999 in item or 255 in item:
                return False

            # Check if the shape of the tile matches the expected shape
            if item.shape != (self.band_count, self.tile_width, self.tile_height):
                return False

            # Check for specific values in the tile at a specific index
            if np.isin(
                item[7, :, :],
                [352, 368, 392, 416, 432, 480, 840, 864, 880, 904, 928, 944, 1352],
            ).any():
                return False
            
            # Tile is valid if it passes all the checks
            return True

    def get_label(self, dataset_index, row, col):
        """
        Get the label for a specific dataset index, row, and column.

        Args:
            dataset_index (int): The index of the dataset.
            row (int): The row index.
            col (int): The column index.

        Returns:
            int: The label value.

        Raises:
            IndexError: If the row and column indices are out of bounds.

        """
        for image in self.image_datasets[dataset_index]:
            # Get the x and y coordinates in the raster for the given dataset index, row, and column
            (x_raster, y_raster) = image.xy(row, col)

            # Convert the coordinates to the label projection if they are not already in the same projection
            if self.raster_proj != self.label_proj:
                transformer = Transformer.from_crs(
                    self.raster_proj.srs, self.label_proj.srs, always_xy=True
                )
                # pylint: disable=E0633
                x_raster, y_raster = transformer.transform(x_raster, y_raster)

            # Convert the raster coordinates to row and column indices in the label dataset
            row, col = self.label_dataset.index(x_raster, y_raster)

            # Get the label value at the specified indices
            window = ((row, row + 1), (col, col + 1))
            data = LandCoverClassDict().merge_classes(
                self.label_dataset.read(1, window=window, masked=False, boundless=True),
                "landsat",
            )
            label = data[0, 0]

            return label