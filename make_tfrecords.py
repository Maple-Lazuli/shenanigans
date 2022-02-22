import numpy as np
import tensorflow as tf
import os
import shutil
import random
import argparse


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
            label_list.append(np.frombuffer(label, dtype=np.uint8))
            label = file.read(1)
    return np.array(label_list).flatten()


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
        "label": _int64_feature(annotation),
        "image_raw": _bytes_feature(image.flatten().tobytes())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def manage_mnist_datafiles(dir):
    """
    Returns a tuple list of image & label pairs
    """
    return [
        (
            dir + "t10k-images.idx3-ubyte", dir + "t10k-labels.idx1-ubyte"
        ), (
            dir + "train-images.idx3-ubyte", dir + "train-labels.idx1-ubyte"
        )
    ]


def make_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def build_tf_dataset(images_file, labels_file):
    """
    Builds the dataset from the passed files
    Parameters
    ----------
    images_file: The file that contains the images
    labels_file: The file that contains the labels
    Returns
    -------
    Returns a list of TF Examples
    """
    images = read_images(images_file)
    labels = read_labels(labels_file)
    return [(image, label) for image, label in zip(images, labels)]


def partition_examples(examples, splits_dict, shuffle=True, seed=101011):
    # TODO: Print examples here and see if they are sequential...
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


def create_tfrecords(data_dir,
                     output_dir,
                     tfrecords_name,
                     datapath_to_examples_fn=build_tf_dataset,
                     tf_example_builder_fn=build_tf_example,
                     partition_examples_fn=partition_examples,
                     splits_dict={"data": 1.0}):
    # gross solution
    batch1, batch2 = manage_mnist_datafiles(data_dir)
    examples = datapath_to_examples_fn(batch1[0], batch1[1])
    examples += datapath_to_examples_fn(batch2[0], batch2[1])

    partitioned_examples = partition_examples_fn(examples, splits_dict)

    for (split_name, split_examples) in partitioned_examples.items():

        print(f"Writing partition {split_name} w/ {len(split_examples)} examples")

        # build a clean directory to store the partition in
        partition_output_dir = os.path.join(output_dir, split_name)
        make_clean_dir(partition_output_dir)

        # TODO - ask about the grouping portion

        tfrecord_name = tfrecords_name + "_" + split_name + '.tfrecords'
        output_path = os.path.join(partition_output_dir,
                                   tfrecord_name)

        with tf.io.TFRecordWriter(output_path) as writer:
            for example in split_examples:
                tf_example = tf_example_builder_fn(example)
                writer.write(tf_example.SerializeToString())


def main_cli(flags):
    split_dict = {"train": 0.8, "valid": 0.1, "test": 0.1}

    partition_fn = partition_examples

    create_tfrecords(data_dir=flags.data_dir,
                     output_dir=flags.output_dir,
                     tfrecords_name=flags.name,
                     datapath_to_examples_fn=build_tf_dataset,
                     tf_example_builder_fn=build_tf_example,
                     partition_examples_fn=partition_fn,
                     splits_dict=split_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str,
                        default="mnist",
                        help='Name of the dataset to build.')

    parser.add_argument('--data_dir', type=str,
                        default="mnist/",
                        help='Path to mnist data.')

    parser.add_argument('--output_dir', type=str,
                        default="mnist_tf/",
                        help='Path to the output directory.')

    flags, unparsed = parser.parse_known_args()

    main_cli(flags)
