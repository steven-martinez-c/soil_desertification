"""
This module provides functions for analyzing and visualizing land cover classification data.
"""
import os
import re
import tarfile
import tempfile
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from shapely.geometry import Point
from pyproj import Proj, Transformer
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from src.layers.dict_class import LandCoverClassDict


def samples_by_class(labels, class_name):
    """
    Counts the number of samples for a specific class in a labeled image.

    Args:
        labels (numpy.ndarray): Labeled image where each pixel is assigned a class label.
        class_name (list): List of class names corresponding to the class labels.

    Returns:
        list: A list of tuples containing the class name and the number of samples for each class.

    """
    unique, counts = np.unique(labels, return_counts=True)
    data_list = [(class_name[x], counts) for x, counts in list(zip(unique, counts))]

    return data_list


def percentage_by_class(labels, class_names):
    """
    Calculates the percentage of samples for each class in a labeled image.

    Args:
        labels (numpy.ndarray): Labeled image where each pixel is assigned a class label.
        class_names (list): List of class names corresponding to the class labels.

    """
    unique, counts = np.unique(labels, return_counts=True)
    unique, counts = unique[1:], counts[1:]  # skip 0
    prop_list = []
    for row, col in enumerate(unique):
        prop_list.append(counts[row] / np.sum(counts))
        print(f"{class_names[col]}: {counts[row] / np.sum(counts):.2%}")


def distribution_of_land(data):
    """
    Plot the distribution of land cover types on a logarithmic scale.

    Args:
        data (list): List of tuples containing land cover types and their frequencies.

    """
    plt.figure(figsize=(1.62 * 6, 6))
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.xlabel("Land Cover Type")
    plt.ylabel("Log(Frequency)")
    plt.title("Distribution of Land Cover Types - Logarithmic Scale")
    _ = plt.bar([classe[0] for classe in data], [classe[1] for classe in data])


def plot_merge_class(labels, colors, class_names):
    """
    Plots the merged classes of a labeled image using false colors.

    Args:
        labels (numpy.ndarray): Labeled image where each pixel is assigned a class label.
        colors (dict): Dictionary mapping class labels to RGB color values.
        class_names (list): List of class names corresponding to the class labels.

    """
    numbers = labels.max() + 1

    # Convert colors from 0-255 to 0-1
    for key, value in colors.items():
        colors[key] = [color / 255.0 for color in value]

    index_false_colors = [colors[key] for key in range(numbers)]

    cmap_false_colors = plt.matplotlib.colors.ListedColormap(
        index_false_colors, "Classification", numbers
    )

    patches = [
        mpatches.Patch(
            color=cmap_false_colors.colors[class_id], label=class_names[class_id]
        )
        for class_id in range(numbers)
    ]

    _, axs = plt.subplots(figsize=(10, 10))

    # Plot the labeled image
    _ = axs.imshow(labels[0, :, :], cmap=cmap_false_colors, interpolation="none")

    axs.legend(handles=patches)
    axs.set_title("Classes")

    plt.show()


def process_data_points(train_pixels, landsat_datasets, dataset_labels):
    """
    Process the given data points to generate training and validation pixel locations.

    Args:
        train_pixels (list): A list of training pixel locations.
        landsat_datasets (list): A list of Landsat datasets.
        dataset_labels (array): The dataset labels.

    Returns:
        numpy.array: The label locations.
    """
    # generate the training and validation pixel locations
    all_labels = []
    label_locations = []
    progression_bar = tqdm(train_pixels)
    class_names = LandCoverClassDict().get_landsat_dictionary()

    for index, pixel in enumerate(progression_bar):
        progression_bar.set_description(f"Processing data point: {index}")

        # row, col location in landsat
        row, col = pixel[0]
        ds_index = pixel[1]
        l8_proj = Proj(landsat_datasets[ds_index].crs)
        label_proj = Proj(dataset_labels.crs)
        transformer = Transformer.from_crs(l8_proj.srs, label_proj.srs, always_xy=True)

        # localización geográfica
        x, y = landsat_datasets[ds_index].xy(row, col)
        # pasar de la proyección label a la proyección del mosaico
        x, y = transformer.transform(x, y)  # pylint: disable=E0633
        # obtener la posición de fila y columna en la etiqueta
        row, col = dataset_labels.index(x, y)
        label_locations.append([row, col])

        # formato (bandas, altura, anchura)
        window = ((row, row + 1), (col, col + 1))
        data = LandCoverClassDict().merge_classes(
            dataset_labels.read(1, window=window, masked=False, boundless=True),
            "landsat",
        )
        all_labels.append(data[0, 0])

    label_locations = np.array(label_locations)

    unique, counts = np.unique(np.array(all_labels), return_counts=True)
    points = [(class_names[x], counts) for x, counts in list(zip(unique, counts))]
    print("\n".join(map(str, points)))

    return label_locations


def locations_of_pixels(labels, locations, colors):
    """
    Plots the locations of training pixels on the image.

    Args:
        labels (ndarray): Array containing the labels of the dataset.
        locations (ndarray): Array containing the locations of the training pixels.
        colors (dict): Dictionary mapping class labels to colors.

    """
    num_classes = int(np.max(labels)) + 1
    index_false_colors = [colors[key] for key in range(0, num_classes)]
    cmap_false_colors = plt.matplotlib.colors.ListedColormap(
        index_false_colors, "Classification", num_classes
    )

    _, axs = plt.subplots(figsize=(1.62 * 10, 10))

    # Plot the dataset labels with false colors
    _ = axs.imshow(labels[0, :, :], cmap=cmap_false_colors, interpolation="none")

    # Scatter plot the locations of training pixels
    axs.scatter(locations[:, 1], locations[:, 0], s=10, c="r")
    axs.set_title("Locations of the training pixels")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_dict, title=None, normalize=False):
    """
    Print and plot the confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_dict (dict): Dictionary mapping class identifiers to class names.
        normalize (bool, optional): Indicates whether to normalize the confusion matrix.
        title (str, optional): Title of the plot. Default is None.

    Returns:
        ax (matplotlib Axes): Axes of the plot.
    """

    if not title:
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    # Calculate the confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    classes = np.array(list(class_dict))

    # Use only the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # Convert class identifiers to class names using the class dictionary
    class_names = [class_dict[cover_class] for cover_class in classes]

    if normalize:
        matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    fig, axis = plt.subplots(figsize=(10, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="viridis")
    axis.figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate x-axis labels and adjust their alignment
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over the dimensions of the confusion matrix and create text annotations
    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                format(matrix[row, col], fmt),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > thresh else "black",
            )
    fig.tight_layout()
    return axis


def extract_landsat_images(path_tar):
    """
    Decompresses images from a Landsat TAR file.

    Args:
        path_tar (str): The path to the TAR file that contains the Landsat images.

    Returns:
        tuple: A tuple containing two lists. The first list contains the paths to the
        decompressed bands of the Landsat images. The second list contains the paths
        to the decompressed metadata files.
    """
    base_dir = "/tmp/landsat/"
    os.makedirs(base_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=base_dir)

    metadata = ""
    bands_dataset = []
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "QA_PIXEL", "BQA"]

    with tarfile.open(path_tar, "r") as tar:
        for name in tar.getnames():
            if name.endswith(tuple(f"_{band}.TIF" for band in bands)):
                bands_dataset.append(f"{tmp}/{name}")
                tar.extract(name, path=tmp)
            elif name.endswith("MTL.txt"):
                metadata = f"{tmp}/{name}"
                tar.extract(name, path=tmp)

    return bands_dataset, metadata


def read_metadata(path_mtl):
    """
    Read metadata from a txt file.

    Parameters:
        ruta_metadatos (str): The path to the metadata file.

    Returns:
        str: The metadata read from the file, or None if there was an error.
    """
    try:
        with open(path_mtl, "r", encoding="utf-8") as file:
            metadata = file.read()

        return metadata

    except Exception as error:
        print(f"Error al leer los metadatos: {str(error)}")


def search_mtl_params(param, metadata):
    """
    Search for a specific parameter in metadata and return its value.

    Args:
        param (str): The name of the parameter to search for in the metadata file.
        metadata (str): The content of the metadata file in string format.

    Returns:
        str or None: The value of the parameter if found, or None if not found.
    """
    # Try to capture numeric values (decimal or scientific notation)
    match_numeric = re.search(rf"{param} = (-?[\d\.]+(?:[Ee][-+]?\d+)?)", metadata)
    if match_numeric:
        return match_numeric.group(1)

    # Try to capture text values
    match_text = re.search(rf'{param} = "(.*?)"', metadata)
    if match_text:
        return match_text.group(1)

    # If no value is found, return None
    return None


def save_pixels(data, out_path):
    """
    Save the given pixel data to a file.

    Args:
        data (list): A list of tuples containing pixel coordinates and their corresponding values.
        out_path (str): The file path where the output file will be saved.
    """
    # Extract coordinates and values from the data
    coordinates = [(x, y) for (x, y), _ in data]
    values = [value for _, value in data]

    # Create a GeoDataFrame with coordinates and values
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(x, y) for x, y in coordinates],
            "index": values,
            "row": [x for x, _ in coordinates],
            "col": [y for _, y in coordinates],
        }
    )

    # Save the GeoDataFrame to the specified output path
    gdf.to_file(out_path)


def cloud_masking(value):
    """
    Determines if a pixel is a cloud based on its bitmask value.

    Args:
        value (int): The bitmask value of the pixel.

    Returns:
        bool: True if the pixel is a cloud, False otherwise.

    Cloud bitmask:

    * Simple bits:
        * bit 0: fill
        * bit 1: dilated cloud
        * bit 2: cirrus high
        * bit 3: cloud
        * bit 4: cloud shadow

    * Pairs of bits:
        * bit 8 and 9: cloud confidence (high)
        * bit 10 and 11: cloud shadow confidence (high)
        * bit 14 and 15: cirrus confidence (high)
    """

    # Check if any of the simple bits are set.
    cloud_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)

    # Check if any of the pair of bits are set.
    confidence_bits = (
        ((1 << 8) & (1 << 9))
        | ((1 << 8) & (0 << 9))
        | ((1 << 10) & (1 << 11))
        | ((1 << 10) & (0 << 11))
        | ((1 << 14) & (1 << 15))
        | ((1 << 14) & (0 << 15))
    )

    return (value & cloud_bits) != 0 or (value & confidence_bits) != 0
