import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

class_names_s2 = dict((
    (0, 'Sin Clasificar'),
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


merge_class_names_s2 = dict((
    (0, 'Sin Clasificar'),
    (1, 'Agua'),
    (2, 'Arboles'),
    (3, 'Area Construida'),
    (4, 'Suelo desnudo'),
    (5, 'Cultivos'),
    (6, 'Pastizales')
))

colors_s2 = dict((
    (0, (0, 0, 0)),  # Background
    (1, (0, 0, 251)),  # Agua
    (2, (62, 178, 49)),  # Arboles
    (3, (245, 106, 0)),  # Cultivos
    (4, (255, 0, 0)),  # Area Construida
    (5, (122, 125, 74)),  # Suelo desnudo
    (6, (255, 165, 81)),  # Pastizales
))

reclassification_s2 = {
    # Change 255 to 0
    255: 0,
    10: 0,
    9: 0,
    # Classify Area Construida
    7: 3,
    # Classify cultivos
    4: 5,
    # Classify Suelo desnudo
    8: 4,
    # Classify pastures
    11: 6,
}


def open_raster(raster):
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
    dataset = rio.open(raster, dtype='uint8')
    num_bands = dataset.count
    print('Number of bands in the image: {n}\n'.format(n=num_bands))

    rows, cols = dataset.shape
    print('Image size: {r} rows x {c} columns\n'.format(
        r=rows, c=cols))

    driver = dataset.driver
    print('Raster driver: {d}\n'.format(d=driver))

    proj = dataset.crs
    print('Image projection:')
    print(proj)

    return dataset


def read_raster(raster, window=None):
    """
    Reads data from a raster file.

    Args:
        raster (rasterio.DatasetReader): The raster dataset.
        window (optional): A window to specify the subset of the raster to read. Defaults to None.

    Returns:
        numpy.ndarray: The data read from the raster.

    """
    if window:
        return raster.read(window=window)
    return raster.read()


def samples_by_class(labels_image, class_name):
    """
    Counts the number of samples for a specific class in a labeled image.

    Args:
        labels_image (numpy.ndarray): Labeled image where each pixel is assigned a class label.
        class_name (list): List of class names corresponding to the class labels.

    Returns:
        list: A list of tuples containing the class name and the number of samples for each class.

    """
    unique, counts = np.unique(labels_image, return_counts=True)
    data_list = [(class_name[x], counts)
                 for x, counts in list(zip(unique, counts))]

    return data_list


def percentage_by_class(labels_image, class_names):
    """
    Calculates the percentage of samples for each class in a labeled image.

    Args:
        labels_image (numpy.ndarray): Labeled image where each pixel is assigned a class label.
        class_names (list): List of class names corresponding to the class labels.

    Returns:
        None

    Prints:
        The percentage of samples for each class in the labeled image.

    """
    unique, counts = np.unique(labels_image, return_counts=True)
    unique, counts = unique[1:], counts[1:]  # skip 0
    prop_list = []
    for ii, jj in enumerate(unique):
        prop_list.append(counts[ii] / np.sum(counts))
        print(str(class_names[jj] + ': '),
              '{:.2f}'.format(counts[ii] / np.sum(counts)))


def distribution_of_land(data):
    """
    Plot the distribution of land cover types on a logarithmic scale.

    Args:
        data (list): List of tuples containing land cover types and their frequencies.

    Returns:
        None
    """
    plt.figure(figsize=(1.62 * 6, 6))
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.xlabel("Land Cover Type")
    plt.ylabel("Log(Frequency)")
    plt.title("Distribution of Land Cover Types - Logarithmic Scale")
    _ = plt.bar([classe[0] for classe in data], [classe[1] for classe in data])


def plot_merge_class(labels_image, colors, class_names):
    """
    Plots the merged classes of a labeled image using false colors.

    Args:
        labels_image (numpy.ndarray): Labeled image where each pixel is assigned a class label.
        colors (dict): Dictionary mapping class labels to RGB color values.
        class_names (list): List of class names corresponding to the class labels.

    Returns:
        None

    Displays:
        Plot of the merged classes with false colors.

    """
    numbers = labels_image.max() + 1

    # Convert colors from 0-255 to 0-1
    for k, v in colors.items():
        colors[k] = [c / 255.0 for c in v]

    index_false_colors = [colors[key] for key in range(numbers)]

    cmap_false_colors = plt.matplotlib.colors.ListedColormap(
        index_false_colors, 'Classification', numbers)

    patches = [mpatches.Patch(color=cmap_false_colors.colors[class_id], label=class_names[class_id])
               for class_id in range(numbers)]

    _, axs = plt.subplots(figsize=(10, 10))

    # Plot the labeled image
    _ = axs.imshow(labels_image[0, :, :],
                   cmap=cmap_false_colors,
                   interpolation='none')

    axs.legend(handles=patches)
    axs.set_title('Classes')

    plt.show()


def locations_of_pixels(dataset_labels, label_locations, colors):
    """
    Plots the locations of training pixels on the image.

    Args:
        dataset_labels (ndarray): Array containing the labels of the dataset.
        label_locations (ndarray): Array containing the locations of the training pixels.
        colors (dict): Dictionary mapping class labels to colors.

    Returns:
        None
    """
    num_classes = int(np.max(dataset_labels)) + 1
    index_false_colors = [colors[key] for key in range(0, num_classes)]
    cmap_false_colors = plt.matplotlib.colors.ListedColormap(
        index_false_colors, 'Classification', num_classes)

    _, axs = plt.subplots(figsize=(1.62 * 10, 10))

    # Plot the dataset labels with false colors
    _ = axs.imshow(dataset_labels[0, :, :],
                   cmap=cmap_false_colors,
                   interpolation='none')

    # Scatter plot the locations of training pixels
    axs.scatter(label_locations[:, 1], label_locations[:, 0], s=10, c='r')

    axs.set_title("Locations of the training pixels")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, class_dict, title=None, normalize=False):
    """
    Print and plot the confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (array-like): Class labels.
        class_dict (dict): Dictionary mapping class identifiers to class names.
        normalize (bool, optional): Indicates whether to normalize the confusion matrix.
        title (str, optional): Title of the plot. Default is None.
        cmap (matplotlib colormap, optional): Color map to use. Default is plt.cm.Blues.

    Returns:
        ax (matplotlib Axes): Axes of the plot.
    """
    if not title:
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'

    # Calculate the confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    # Use only the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # Convert class identifiers to class names using the class dictionary
    class_names = [class_dict[cover_class] for cover_class in classes]

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    fig, axis = plt.subplots(figsize=(10, 10))
    image = axis.imshow(matrix, interpolation='nearest', cmap='viridis')
    axis.figure.colorbar(image, ax=axis)

    axis.set(xticks=np.arange(matrix.shape[1]),
             yticks=np.arange(matrix.shape[0]),
             xticklabels=class_names, yticklabels=class_names,
             title=title,
             ylabel='True Label',
             xlabel='Predicted Label')

    # Rotate x-axis labels and adjust their alignment
    plt.setp(axis.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over the dimensions of the confusion matrix and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, format(matrix[i, j], fmt),
                      ha="center", va="center",
                      color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return axis
