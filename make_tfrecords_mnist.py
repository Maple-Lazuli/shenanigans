import numpy as np
import tensorflow as tf
import random
import os


def map_label_to_name(label):
    return str(label)


def read_images(file_name):
    """
    Reads from the non_standard mnist dataset to extract images
    Parameters
    ----------
    file_name: The full filename to read.

    Returns
    -------
    list of images as np.ndarray
    """
    image_list = []
    with open(file_name, 'rb') as file:
        file.read(16)  # skip header bytes
        data = file.read(784)
        while len(data) == 784:
            image_list.append(np.frombuffer(data, dtype=np.uint8).reshape(28, 28))
            data = file.read(784)
    return np.array(image_list)


def read_labels(file_name):
    """
    Reads from the mnist labels dataset. One hot encodes the labels
    Parameters
    ----------
    file_name: The name of the file to read labels from

    Returns
    -------
    returns numpy ndarray
    """
    label_list = []
    with open(file_name, 'rb') as file:
        file.read(8)  # skip header bytes
        label = file.read(1)
        while len(label) == 1:
            label_ohe = np.zeros(10, dtype=np.int)
            label_ohe[np.frombuffer(label, dtype=np.uint8)] = 1
            label_list.append(label_ohe)
            label = file.read(1)
    return np.array(label_list)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_tf_example(example):
    """
    Creates a TF Example from the image and the annotation
    Parameters
    ----------
    image: The image to be added as a feature
    annotation: the class of the image

    Returns
    -------
    TF Example
    """
    image, annotation = example

    feature = {
        "height": _int64_feature(image.shape[0]),
        "width": _int64_feature(image.shape[1]),
        "depth": _int64_feature(0) if len(image.shape) < 3 else _int64_feature(image.shape[2]),
        "label": _bytes_feature(annotation.tobytes()),
        "image_raw": _bytes_feature(image.flatten().tobytes())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def build_tf_dataset(directory):
    """
    Iterates through the directory to find the MNIST image and labels files. Then parses those images and labels to
    return a tuple of (image,label) pairs

    Parameters
    ----------
    directory: The directory to iterate through

    Returns
    -------
    Returns a list of TF Examples
    """

    file_list = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(subdir, file))

    for file in file_list:
        if file.find("t10k-images.idx3-ubyte") != -1:
            test_images = read_images(file)
        elif file.find("t10k-labels.idx1-ubyte") != -1:
            test_labels = read_labels(file)
        elif file.find("train-images.idx3-ubyte") != -1:
            train_images = read_images(file)
        elif file.find("train-labels.idx1-ubyte") != -1:
            train_labels = read_labels(file)

    train_pairs = [(image, label) for image, label in zip(train_images, train_labels)]
    test_pairs = [(image, label) for image, label in zip(test_images, test_labels)]

    # combine the two data sets to later split into train, test, validate sets
    return train_pairs + test_pairs


def partition_examples(examples, splits_dict, shuffle=True, seed=101011):
    if shuffle:
        random.seed(seed)
        random.shuffle(examples)

    partitions = dict()
    # Store the total number of examples.
    num_examples = len(examples)
    # Iterate over the items specifying the partitions.
    for (split_name, split_fraction) in splits_dict.items():
        # Compute the size of this partition.
        num_split_examples = int(split_fraction * num_examples)
        # Pop the next partition elements.
        partitioned_examples = examples[:num_split_examples]
        examples = examples[num_split_examples:]
        # Map this partitions list of examples to this partition name.
        partitions[split_name] = partitioned_examples

    return partitions
