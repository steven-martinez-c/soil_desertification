"""
    A class that represents a dictionary
"""


class LandCoverClassDict:
    """
    A class that represents a dictionary of land cover classes.
    """

    unknown = 'Sin Clasificar'

    def get_dictionaries(self):
        """
        Returns a dictionary containing the different dictionaries used in the class.
        """
        return {
            "landsat": self.landsat,
            "sentinel": self.sentinel,
            "merged_sentinel_class": self.merged_sentinel_class,
            "colors": self.colors
        }

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

        if satellite == 'sentinel':
            reclassification = {
                255: 0,
                10: 0,
                9: 0,
                7: 3,
                4: 5,
                8: 4,
                11: 6,
            }
        elif satellite == 'landsat':
            reclassification = {
                255: 0
            }

        for index, item in reclassification.items():
            value[value == index] = item

        return value

    landsat = dict((
        (0, unknown),
        (1, 'Bosque'),
        (2, 'Cuerpo de Agua'),
        (3, 'Otras Tierras'),
        (4, 'Tierra Agropecuaria'),
        (5, 'Vegetacion Arbustiva y Herbacea'),
        (6, 'Zona Antropica'),
    ))

    sentinel = dict((
        (0, unknown),
        (1, 'Agua'),
        (2, 'Árboles'),
        (4, 'Vegetación inundada'),
        (5, 'Cultivos'),
        (7, 'Área Construida'),
        (8, 'Suelo desnudo'),
        (9, 'Nieve/hielo'),
        (10, 'Nubes'),
        (11, 'Pastizales')
    ))

    merged_sentinel_class = dict((
        (0, unknown),
        (1, 'Agua'),
        (2, 'Árboles'),
        (4, 'Vegetación inundada'),
        (5, 'Cultivos'),
        (7, 'Área Construida'),
        (8, 'Suelo desnudo'),
        (9, 'Nieve/hielo'),
        (10, 'Nubes'),
        (11, 'Pastizales')
    ))

    colors = {
        0: (0, 0, 0),       # Background (Negro)
        1: (0, 0, 251),     # Agua (Azul)
        2: (62, 178, 49),   # Árboles (Verde)
        3: (245, 106, 0),   # Cultivos (Naranja)
        4: (255, 0, 0),     # Área Construida (Rojo)
        5: (122, 125, 74),  # Suelo desnudo (Marrón)
        6: (255, 165, 81)   # Pastizales (Melocotón)
    }
