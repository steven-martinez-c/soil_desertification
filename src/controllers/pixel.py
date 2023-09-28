"""
    PixelGenerator
"""

import random
import math
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
from rasterio.windows import Window
from pyproj import Proj, Transformer
from src.controllers.raster import RasterController
from src.layers.dict_class import LandCoverClassDict


class PixelController:
    """
    A class that controls pixels.

    This class provides methods for converting coordinates, working with labels, and more.

    Parameters:
        label_dataset (Dataset): The label dataset to work with.

    Attributes:
        label_dataset (Dataset): The label dataset being used.
        label_proj (Proj): The projection of the label dataset.
    """

    def __init__(self, label_dataset):
        self.label_dataset = label_dataset
        self.label_proj = Proj(label_dataset.crs)
        self.raster = RasterController()
        self.dict = LandCoverClassDict()

    def _convert_coordinates(self, image_dataset, coords):
        """
        Convert a list of coordinates from one coordinate system to another.

        Parameters:
            image_dataset (Dataset): The dataset containing the image coordinates.
            coords (List[Tuple[float, float]]): The list of coordinates to be converted.

        Returns:
            List[Tuple[float, float]]: The list of converted coordinates.
        """
        l8_proj = Proj(image_dataset.crs)
        converted_coords = []

        transformer = Transformer.from_crs(
            l8_proj.srs, self.label_proj.srs, always_xy=True
        )
        for x, y in coords:
            x, y = transformer.transform(x, y)  # pylint: disable=E0633
            row, col = self.label_dataset.index(x, y)
            converted_coords.append((col, row))  # Swapping col and row

        return converted_coords

    def _get_raster_polygon(self, image_dataset):
        """
        Generate a polygon representing the raster bounds of the given image dataset.

        Parameters:
            image_dataset (ImageDataset): The image dataset containing the raster data.

        Returns:
            Polygon: A polygon representing the raster bounds of the image dataset.
        """
        raster_points = (
            image_dataset.transform * (0, 0),
            image_dataset.transform * (image_dataset.width, 0),
            image_dataset.transform *
            (image_dataset.width, image_dataset.height),
            image_dataset.transform * (0, image_dataset.height),
        )
        new_raster_points = self._convert_coordinates(
            image_dataset, raster_points)
        return Polygon(new_raster_points)

    def generate_training_pixels(self, image_datasets, train_count, merge=False):
        """
        Generates training pixels for machine learning models.

        Args:
            image_datasets (list): List of image datasets.
            train_count (int): Total number of training pixels to generate.
            merge (bool, optional): Whether to merge classes. Defaults to False.

        Returns:
            list: List of training pixels with their corresponding dataset index.

        """
        train_pixels = []
        labels_image = self.raster.read_raster(self.label_dataset)
        l8_proj = Proj(image_datasets[0].crs)
        train_count_per_dataset = math.ceil(train_count / len(image_datasets))
        print("Train count per dataset: ", train_count_per_dataset)
        points_per_class = train_count_per_dataset // len(
            np.unique(self.dict.merge_classes(labels_image, "landsat"))
        )
        print("Points per class: ", points_per_class)

        for index, image_dataset in enumerate(image_datasets):
            raster_poly = self._get_raster_polygon(image_dataset)
            masked_label_image = self.label_dataset.read(
                window=Window.from_slices(
                    (int(raster_poly.bounds[1]), int(raster_poly.bounds[3])),
                    (int(raster_poly.bounds[0]), int(raster_poly.bounds[2])),
                )
            )

            if merge:
                masked_label_image = self.dict.merge_classes(
                    masked_label_image, "landsat"
                )

            all_points_per_image = []
            progression_bar = tqdm(
                np.unique(self.dict.merge_classes(labels_image, "landsat"))
            )

            for cls in progression_bar:
                class_name = self.dict.get_landsat_dictionary()[int(cls)]
                progression_bar.set_description(
                    f"Processing << {class_name} >>")
                cls = int(cls)
                rows, cols = np.where(masked_label_image[0] == cls)
                all_locations = list(zip(rows, cols))
                random.shuffle(all_locations)

                l8_points = []
                if len(all_locations) != 0:
                    transformer = Transformer.from_crs(
                        self.label_proj.srs, l8_proj.srs, always_xy=True
                    )
                    for row, col in all_locations[:points_per_class]:
                        x, y = self.label_dataset.xy(
                            row +
                            raster_poly.bounds[1], col + raster_poly.bounds[0]
                        )
                        x, y = transformer.transform(x, y)  # pylint: disable=E0633
                        row, col = image_dataset.index(x, y)
                        l8_points.append((row, col))

                    all_points_per_image += l8_points

            dataset_index_list = [index] * len(all_points_per_image)
            dataset_pixels = list(
                zip(all_points_per_image, dataset_index_list))
            train_pixels += dataset_pixels
            random.shuffle(train_pixels)
            
        return train_pixels
