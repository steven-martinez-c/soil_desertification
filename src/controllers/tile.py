"""
    This class is responsible for generating tiles from image datasets and label datasets.
"""
import math
import numpy as np
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from pyproj import Proj, Transformer
from src.layers.dict_class import LandCoverClassDict


class TileController:
    """
    This class is responsible for generating tiles from image datasets and label datasets.
    """

    def __init__(self, tile_height, tile_width, pixel_locations, batch_size):
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.batch_size = batch_size
        self.pixel_locations = pixel_locations

    def _get_tile(self, dataset, row, col, band_count, buffer):
        """
        Obtiene un mosaico de un dataset de imagen.

        Args:
            dataset: El dataset de imagen.
            row: La fila del mosaico.
            col: La columna del mosaico.
            band_count: El número de bandas del dataset de imagen.
            buffer: El tamaño del búfer.

        Returns:
            El mosaico.
        """
        tile = dataset.read(
            list(np.arange(1, band_count + 1)),
            window=Window(
                col - buffer, row - buffer, self.tile_width, self.tile_height
            ),
        )
        return tile

    def _get_label(self, rasters, label_dataset, row, col, index):
        """
        Obtiene la etiqueta de un píxel.

        Args:
            rasters: La lista de datasets de imagen.
            label_dataset: El dataset de etiquetas.
            row: La fila del píxel.
            col: La columna del píxel.
            index: El índice del dataset de imagen.

        Returns:
            La etiqueta del píxel.
        """

        l8_proj = Proj(rasters[0].crs)
        label_proj = Proj(label_dataset.crs)

        (x, y) = rasters[index].xy(row, col)

        if l8_proj != label_proj:
            transformer = Transformer.from_crs(
                l8_proj.srs, label_proj.srs, always_xy=True
            )
            x, y = transformer.transform(x, y)  # pylint: disable=E0633

        row, col = label_dataset.index(x, y)

        window = ((row, row + 1), (col, col + 1))
        data = LandCoverClassDict().merge_classes(
            label_dataset.read(1, window=window, masked=False, boundless=True),
            "landsat",
        )
        label = data[0, 0]

        if label == 0 or np.isnan(label).any():
            pass

        return label

    def _is_valid_tile(self, tile, band_count):
        """
        Verifica si un mosaico es válido.

        Args:
            tile: El mosaico.
            band_count: El número de bandas del mosaico.

        Returns:
            `True` si el mosaico es válido, `False` de lo contrario.
        """

        return (
            tile.size > 0
            and np.amax(tile) > 0
            and not np.isnan(tile).any()
            and -9999 not in tile
            and tile.shape == (band_count, self.tile_width, self.tile_height)
        )

    def tile_generators(self, images_datasets, label_dataset):
        """
        Generates batches of image and label data for training a model.

        Args:
            images_datasets (list): A list of image datasets.
            label_dataset (str): The label dataset.

        Returns:
            generator: A generator that yields batches of image and label data.
        """
        col = row = i = 0
        band_count = images_datasets[0].count
        class_count = len(LandCoverClassDict().get_landsat_dictionary())
        buffer = math.ceil(self.tile_height / 2)

        while True:
            image_batch = np.zeros(
                (self.batch_size, self.tile_height, self.tile_width, band_count - 1)
            )
            label_batch = np.zeros((self.batch_size, class_count))
            bath_size = 0
            while bath_size < self.batch_size:
                # if we're at the end  of the data just restart
                if i >= len(self.pixel_locations):
                    i = 0
                row, col = self.pixel_locations[i][0]
                dataset_index = self.pixel_locations[i][1]
                i += 1

                tile = self._get_tile(
                    images_datasets[dataset_index], row, col, band_count, buffer
                )
                if not self._is_valid_tile(tile, band_count):
                    continue
                tile = tile[0:7]
                reshaped_tile = (reshape_as_image(tile) - 982.5) / 1076.5
                label = self._get_label(
                    images_datasets, label_dataset, row, col, dataset_index
                )
                if label == 0 or np.isnan(label).any():
                    continue
                label_batch[bath_size][label] = 1
                image_batch[bath_size] = reshaped_tile
                bath_size += 1
            yield (image_batch, label_batch)
