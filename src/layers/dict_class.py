"""
    A class that represents a dictionary
"""
import numpy as np


class LandCoverClassDict:
    """
    A class that represents a dictionary of land cover classes.
    """

    unknown = "Sin Clasificar"

    def get_landsat_dictionary(self):
        """
        Returns the "landsat" dictionary.

        Returns:
            dict: The "landsat" dictionary.
        """
        return self.landsat

    def get_sentinel_dictionary(self):
        """
        Returns the "sentinel" dictionary.

        Returns:
            dict: The "sentinel" dictionary.
        """
        return self.sentinel

    def get_merged_sentinel_class_dictionary(self):
        """
        Returns the "merged_sentinel_class" dictionary.

        Returns:
            dict: The "merged_sentinel_class" dictionary.
        """
        return self.merged_sentinel_class

    def get_colors_dictionary(self):
        """
        Returns the "colors" dictionary.

        Returns:
            dict: The "colors" dictionary.
        """
        return self.colors

    def merge_classes(self, value, satellite):
        """
        Merge classes in the given value array based on the specified satellite.

        Parameters:
            value (ndarray): The input array to be modified.
            satellite (str): The satellite type. Can be 'sentinel' or 'landsat'.

        Returns:
            ndarray: The modified input array with merged classes.
        """
        reclassification = {}

        if satellite == "sentinel":
            reclassification = {
                255: 0,
                10: 0,
                9: 0,
                7: 3,
                4: 5,
                8: 4,
                11: 6,
            }
        elif satellite == "landsat":
            reclassification = {255: 0}

        # Utiliza np.vectorize para aplicar la reasignación a todos los elementos del array
        reclassify = np.vectorize(lambda x: reclassification.get(x, x))
        value = reclassify(value)

        return value

    class_to_index = dict(
        (
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
        )
    )

    landsat = dict(
        (
            (0, unknown),
            (1, "Bosque"),
            (2, "Cuerpo de Agua"),
            (3, "Otras Tierras"),
            (4, "Tierra Agropecuaria"),
            (5, "Vegetacion Arbustiva y Herbacea"),
            (6, "Zona Antropica"),
        )
    )

    sentinel = dict(
        (
            (0, unknown),
            (1, "Agua"),
            (2, "Árboles"),
            (4, "Vegetación inundada"),
            (5, "Cultivos"),
            (7, "Área Construida"),
            (8, "Suelo desnudo"),
            (9, "Nieve/hielo"),
            (10, "Nubes"),
            (11, "Pastizales"),
        )
    )

    merged_sentinel_class = dict(
        (
            (0, unknown),
            (1, "Agua"),
            (2, "Árboles"),
            (4, "Vegetación inundada"),
            (5, "Cultivos"),
            (7, "Área Construida"),
            (8, "Suelo desnudo"),
            (9, "Nieve/hielo"),
            (10, "Nubes"),
            (11, "Pastizales"),
        )
    )

    colors = dict(
        (
            (0, (0, 0, 0)),  # Background
            (1, (62, 178, 49)),  # Árboles (Verde)
            (2, (0, 0, 251)),  # Agua (Azul)
            (3, (122, 125, 74)),  # Suelo desnudo (Marrón)
            (4, (245, 106, 0)),  # Cultivos (Naranja)
            (5, (255, 165, 81)),  # Pastizales (Melocotón)
            (6, (255, 0, 0)),  # Área Construida (Rojo)
        )
    )
