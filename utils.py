"""
Contains Functions To Extract MNIST Images and Labels
"""
import numpy as np


def read_mnist_images(file_name):
    """
    Reads from the non_standard mnist dataset to extract images
    :param file_name: The name of the image file to read from
    :return: Returns np.ndarray with shape (, 784)
    """
    image_list = []
    with open(file_name, 'rb') as file:
        file.read(16)  # skip header bytes
        data = file.read(784)
        while len(data) == 784:
            image_list.append(np.frombuffer(data, dtype=np.uint8))
            data = file.read(784)
    return np.array(image_list)


def read_mnist_labels(file_name):
    """
    Reads from the mnist labels dataset. One hot encodes the labels
    :param file_name: The name of the label file to read from
    :return: np.ndarray with shape (, 10)
    """
    label_list = []
    with open(file_name, 'rb') as file:
        file.read(8)  # skip header bytes
        label = file.read(1)
        while len(label) == 1:
            one_hot_encoded = np.zeros(10, dtype=np.uint8)
            one_hot_encoded[np.frombuffer(label, dtype=np.uint8)] = 1
            label_list.append(one_hot_encoded)
            label = file.read(1)
    return np.array(label_list)
