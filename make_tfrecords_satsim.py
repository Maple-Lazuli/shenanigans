import numpy as np
import tensorflow as tf
import os
import json
import random
import astropy.io.fits


def map_label_to_name(label):
    if label == 0:
        return "Nominal"
    elif label == 1:
        return "Collision High"
    elif label == 2:
        return "Collision Low"
    elif label == 3:
        return "RPO"
    elif label == 4:
        return "Breakup"


def get_labels(path):
    """
    The generated data stores the labels for the images in the directory structure and the purpose of this function is
    to extract those labels.

    The numeric class labels are:
    0 - nominal
    1 - collision_high
    2 - collision_low
    3 - rpo
    4 - breakup

    Parameters
    ----------
    path - The path of the image to process

    Returns
    -------
    Tuple for class number, class name, and boolean for stray light
    """
    classes = ["nominal", "collision_high", "collision_low", "rpo", "breakup"]
    for idx, val in enumerate(classes):
        if path.find(val) != -1:
            class_number = idx
            class_name = val
            break

    stray_light = 0 if path.find("no_stray_light") else 1

    return class_number, class_name, stray_light


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def build_tf_example(example):
    """
    Creates a TF Example from the passed example dictionary.
    Parameters
    ----------
    example: a dictionary that contains the path to the fits image, sensor properties, class number, and
    stray_light boolean

    Returns
    -------
    TF Example
    """

    height = example['sensor']['height']
    width = example['sensor']['width']
    field_of_view_x = example['sensor']['iFOVx']
    field_of_view_y = example['sensor']['iFOVy']
    label = np.zeros(5, dtype=np.int)
    label[example['class_number']] = 1
    stray_light = example['has_stray_light']
    class_name = example['class_name']
    try:
        image_raw = astropy.io.fits.getdata(example['fits_image_path']).astype(np.uint16)
    except:
        print("error parsing fits")
        return None

    features = {
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "depth": _int64_feature(16),  # 16 bits per pixel for FITS images
        "field_of_view_x": _floats_feature(field_of_view_x),
        "field_of_view_y": _floats_feature(field_of_view_y),
        "stray_light": _int64_feature(stray_light),
        "class_name": _bytes_feature(class_name.encode()),
        "label": _bytes_feature(label.tobytes()),
        "image_raw": _bytes_feature(image_raw.tobytes())
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def build_tf_dataset(directory):
    """
    Builds the dataset from the passed directory. The function iterates through all subdirectories of the passed
    directory to find the fits and json files.

    Parameters
    ----------
    directory: the directory to iterate through.

    Returns
    -------
    Returns a list of example dictionaries to be added to a tf record.
    """

    # Enumerate all the files in the directory
    file_list = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(subdir, file))

    # Find all the JSONs
    json_list = [e for e in file_list if e[-4:] == 'json']

    # Read the jsons and store them in a dictionary
    # the keys will be made from the directory and base name
    json_dict = dict()
    for j in json_list:
        # cut out the 'annotation' part of the directory
        primary_path = "".join(j.split("/")[0:-2])
        # get the file name and remove the file extension
        file_name = j.split("/")[-1][:-5]
        key_name = primary_path + file_name

        with open(j, "r") as file_in:
            json_dict[key_name] = {'json_path': j, 'json_dump': json.load(file_in)}

    # Find all the fits files.
    fits_list = [e for e in file_list if e[-4:] == 'fits']

    # Use the json dictionary and fits_list to create a list of examples
    examples = []
    for f in fits_list:
        primary_path = "".join(f.split("/")[0:-2])
        file_name = f.split("/")[-1][:-5]
        key_name = primary_path + file_name
        class_number, class_name, stray_light = get_labels(f)

        sensor_data = json_dict[key_name]['json_dump']['data']['sensor']

        example = {
            "fits_image_path": f,
            "sensor": sensor_data,
            "class_number": class_number,
            "class_name": class_name,
            "has_stray_light": stray_light
        }
        examples.append(example)

    return examples


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
