import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import hashlib
from collections import Counter
import argparse
import os
import dataframe_image as dfi
import evaluation_utils as eu
"""
Writes either a training report or a evaluation report. It creates plots showing the datasets, examples, and metrics
"""


def create_roc_plot(tpr, fpr, label_name, write_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fpr, tpr, label=f'label: {label_name}')

    ax.set_title(f"ROC Curve For {label_name} Against Validation Set")
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.legend(loc='best', fontsize=12)

    # Create the save file name
    # use the current time as the seed
    now = datetime.now()
    # use the date hash as the file name
    date_hash = hashlib.md5(str(now).encode())
    image_name = date_hash.hexdigest()

    full_name = f"{write_dir}images/{image_name}.png"
    plt.savefig(full_name)
    plt.close(fig)

    return f"images/{image_name}.png"

def create_image_from_matrix(matrix, write_directory):
    """
    Creates a PNG image from a pandas dataframe
    Parameters
    ----------
    matrix: The pandas dataframe to create an image for
    write_directory: The directory to write the image

    Returns
    -------
    A string with the relative path to the image from the write directory2
    """
    date_hash = hashlib.md5(str(datetime.now()).encode())
    image_name = date_hash.hexdigest()
    full_name = f"{write_directory}images/{image_name}.png"
    dfi.export(matrix, full_name)
    return f"images/{image_name}.png"


def get_dataset_report_examples(dataset, label_key):
    """
    Finds one example for each label for the report.

    Parameters
    ----------
    dataset: The dataset to find examples in
    label_key: the feature key that represents the label

    Returns
    -------
    Returns a list containing one feature for each label
    """
    # setup iterator
    iterator = dataset.get_iterator()
    next_batch = iterator.get_next()

    seen_labels = []
    example_features = []
    dataset_dict = dict()
    with tf.compat.v1.Session() as sess:
        try:
            sess.run(iterator.initializer)
            while True:
                features = sess.run(next_batch)
                for idx, label in enumerate(features[label_key]):
                    label_val = label.argmax()
                    if label_val not in seen_labels:
                        example_dict = dict()
                        for key in features.keys():
                            example_dict[key] = features[key][idx]

                        example_features.append(example_dict)
                        seen_labels.append(label_val)
        except tf.errors.OutOfRangeError:

            return example_features


def create_example_rows(dataset, input_key, label_key, report_location):
    """
    Creates plots based on the input examples, writes them to a subdirectory of the report_location and returns them
    as a string to be added in the report

    Parameters
    ----------
    dataset: The dataset to pull examples from
    input_key: The feature key representing the image
    label_key: The feature key representing the label to find examples for.
    report_location: The location of the report. This function will save the plots to a subdirectory.

    Returns
    -------
    Returns a string encoded in markdown to be added into the plot.
    """
    examples = get_dataset_report_examples(dataset, label_key)
    return_str = ""

    for idx, example in enumerate(examples):
        image = example[input_key].reshape(example['height'], example['width'])
        return_str += f"### Example {idx + 1} \n"
        for idx, key in enumerate(example.keys()):
            if key != input_key:
                return_str += f"{idx + 1}. {key}:{example[key]}\n"
        return_str += f"![image]({create_raster_image(image, report_location)})\n"
    return return_str


def create_raster_image(data, report_location):
    """
    Creates a raster image based on the data and saves the image to the specified location
    Parameters
    ----------
    data: The data to create the raster image from
    report_location: The location of the reports. This function will save the images to a sub directory

    Returns
    -------
    Returns the location of the saved image relative to the reports directory
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(data, cmap='gray')
    ax.grid(False)
    now = datetime.now()
    # use the date hash as the file name
    date_hash = hashlib.md5(str(now).encode())
    image_name = date_hash.hexdigest()
    full_name = f"{report_location}images/{image_name}.png"
    plt.savefig(full_name)
    plt.close(fig)
    return f"images/{image_name}.png"


def get_dataset_metrics(dataset, ignore_list, dataset_value_parser_fn):
    """
    Creates images depicting the number of features in the dataset, such as the number of occurences for each label.

    Parameters
    ----------
    dataset: The dataset to poll numbers for
    ignore_list: features of the dataset to ignore
    dataset_value_parser_fn: function to assist with transforming dataset features to a presentable format

    Returns
    -------
    returns a dictionary with the features
    """
    # setup iterator
    iterator = dataset.get_iterator()
    next_batch = iterator.get_next()

    dataset_dict = dict()
    with tf.compat.v1.Session() as sess:
        try:
            sess.run(iterator.initializer)
            while True:
                features = sess.run(next_batch)

                # iterate through each of the keys
                for feature_key in features.keys():
                    # find the keys not to report
                    if feature_key not in ignore_list:
                        # determine if key is in the dict
                        if feature_key not in dataset_dict.keys():
                            dataset_dict[feature_key] = []

                        # for each value in the batch, add it to the dict
                        for value in features[feature_key]:
                            dataset_dict[feature_key].append(dataset_value_parser_fn(feature_key, value))

        except tf.errors.OutOfRangeError:
            return dataset_dict


def dataset_value_parser(key, value):
    """
    The default parser with no special functionality

    Parameters
    ----------
    key: The feature key
    value: The feature value

    Returns
    -------
    Returns the value passed in without a transformation
    """
    return value


def normalize_dict(d):
    """
    Normalizes the values in the dictionary

    Parameters
    ----------
    d: The dictionary to normalize

    Returns
    -------
    A normalized version of the dictionary
    """
    summ = 0
    for key in d.keys():
        summ += d[key]
    for key in d.keys():
        d[key] = d[key] / summ
    return d


def create_bar_plot(data_set, feature_name, report_location, normalize=False):
    """
    Creates a side by side bar plot for the feature from the two datasets. It expects the datasets to have been passed
    through get_dataset_metrics prior to use.

    Parameters
    ----------
    data_set: The dataset dictionary
    feature_name: The feature to create a bar plot for
    report_location: the location to save the plot in
    normalize: A boolean indicating whether to normalize the data first

    Returns
    -------
    A string representing the location on disk for the image location
    """

    if normalize:
        data_set_counter = normalize_dict(Counter(data_set))
    else:
        data_set_counter = Counter(data_set)

    fig, ax = plt.subplots(figsize=(10, 5))

    x_index = np.arange(len(data_set_counter.keys()))

    labels = []
    values = []
    for key in data_set_counter.keys():
        labels.append(key)
        values.append(data_set_counter[key])

    ax.bar(x_index, values)
    # Plotting

    if normalize:
        ax.set_title(f"Normalized Bar Chart for {feature_name}")
        ax.set_ylabel("Percentage", fontsize=14)
    else:
        ax.set_title(f"Bar Chart for {feature_name}")
        ax.set_ylabel("Occurrences", fontsize=14)

    ax.set_xlabel("Value", fontsize=14)

    ax.set_xticks(x_index)
    ax.set_xticklabels(labels)

    # Finding the best position for legends and putting it
    ax.legend(loc='best', fontsize=12)

    # save the plot
    now = datetime.now()
    # use the date hash as the file name
    date_hash = hashlib.md5(str(now).encode())
    image_name = date_hash.hexdigest()

    full_name = f"{report_location}images/{image_name}.png"
    plt.savefig(full_name)
    plt.close(fig)
    return f"./images/{image_name}.png"


def create_bar_plot_pairs(train_set, validation_set, feature_name, report_location, normalize=False):
    """
    Creates a side by side bar plot for the feature from the two datasets. It expects the datasets to have been passed
    through get_dataset_metrics prior to use.

    Parameters
    ----------
    train_set: The first dataset dictionary
    validation_set: The second dataset dictionary
    feature_name: The feature to create a bar plot for
    report_location: the location to save the plot in
    normalize: A boolean indicating whether to normalize the data first

    Returns
    -------
    A string representing the location on disk for the image location
    """
    if normalize:
        train_counter = normalize_dict(Counter(train_set))
        validation_counter = normalize_dict(Counter(validation_set))
    else:
        train_counter = Counter(train_set)
        validation_counter = Counter(validation_set)

    keys = set()
    [keys.add(key) for key in train_counter.keys()]
    [keys.add(key) for key in validation_counter.keys()]

    N = len(keys)

    train_values = []
    validation_values = []

    for key in keys:
        if key not in train_counter.keys():
            train_values.append(0)
        else:
            train_values.append(train_counter[key])

        if key not in validation_counter.keys():
            validation_values.append(0)
        else:
            validation_values.append(validation_counter[key])

    # Position of bars on x-axis
    ind = np.arange(N)
    width = 0.3
    # Figure size

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(ind, train_values, width, label='Train Set')
    ax.bar(ind + width, validation_values, width, label='Validation Set')
    # Plotting

    if normalize:
        ax.set_title(f"Normalized Bar Chart for {feature_name}")
        ax.set_ylabel("Percentage", fontsize=14)
    else:
        ax.set_title(f"Bar Chart for {feature_name}")
        ax.set_ylabel("Occurrences", fontsize=14)

    ax.set_xlabel("Value", fontsize=14)

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(list(keys))

    # Finding the best position for legends and putting it
    ax.legend(loc='best', fontsize=12)

    # save the plot
    now = datetime.now()
    # use the date hash as the file name
    date_hash = hashlib.md5(str(now).encode())
    image_name = date_hash.hexdigest()

    full_name = f"{report_location}images/{image_name}.png"
    plt.savefig(full_name)
    plt.close(fig)
    return f"./images/{image_name}.png"


class Report:
    def __init__(self):
        self.label_map_fn = None
        self.introduction = None
        self.metrics = []
        self.validation_set = None
        self.train_set = None
        self.write_directory = None
        self.confusion_matrix = None
        self.hyper_parameters = dict()
        self.dataset_value_parser = dataset_value_parser
        self.ignore_list = []
        self.evaluation_metrics = []
        self.evaluation_metric_scalars = []

    def set_introduction(self, introduction):
        self.introduction = introduction

    def add_metric(self, metrics):
        self.metrics.append(metrics)

    def add_evaluatation_metric(self, metric):
        self.evaluation_metrics.append(metric)

    def add_evaluatation_metric_scalar(self, metric):
        self.evaluation_metric_scalars.append(metric)

    def add_hyperparameter(self, param):
        self.hyper_parameters.update(param)

    def set_validation_set(self, validation_set):
        self.validation_set = validation_set

    def set_roc_dict(self, roc_dict):
        self.roc_dict = roc_dict

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_confusion_matrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    def set_write_directory(self, directory):
        self.write_directory = directory

    def set_dataset_value_parser(self, function):
        self.dataset_value_parser = function

    def make_introduction_section(self):
        heading = "# Overview \n"

        introduction = str(self.introduction) + "\n"

        return heading + introduction

    def set_ignore_list(self, ignore_list):
        self.ignore_list = ignore_list

    def set_label_map_fn(self, label_map_fn):
        self.label_map_fn = label_map_fn

    def make_datasets_section(self):
        heading = "# Datasets \n"

        training_heading = "### Training Set \n"

        training_desc = f"The training set located at {self.train_set.get_location()} consists of {self.train_set.get_size()}, served in batch sizes of {self.train_set.get_batch_size()}.\n\n"

        validation_heading = "### Validation Set \n"

        validation_desc = f"The validation set located at {self.validation_set.get_location()} consists of {self.validation_set.get_size()}, served in batch sizes of {self.validation_set.get_batch_size()}.\n\n"

        comparison_heading = "### Validation Set and Training Set Comparison \n"

        comparison_desc = "This section compares the contents of the validation and train sets used.\n"

        train_dict = get_dataset_metrics(self.train_set, ignore_list=self.ignore_list,
                                         dataset_value_parser_fn=self.dataset_value_parser)
        validation_dict = get_dataset_metrics(self.validation_set, ignore_list=self.ignore_list,
                                              dataset_value_parser_fn=self.dataset_value_parser)
        comparisons = ""
        for key in list(train_dict.keys()) + list(validation_dict.keys()):
            comparisons += f"![image]({create_bar_plot_pairs(train_set=train_dict[key], validation_set=validation_dict[key], feature_name=key, report_location=self.write_directory, normalize=False)})\n"
            comparisons += f"![image]({create_bar_plot_pairs(train_set=train_dict[key], validation_set=validation_dict[key], feature_name=key, report_location=self.write_directory, normalize=True)})\n"

        return heading + training_heading + training_desc + validation_heading + validation_desc + comparison_heading + comparison_desc + comparisons

    def make_evaluation_dataset_section(self):
        heading = "# Validation Dataset \n"

        validation_desc = f"The validation dataset located at {self.validation_set.get_location()} consists of {self.validation_set.get_size()}, served in batch sizes of {self.validation_set.get_batch_size()}.\n The charts below depict the distribution of the features of this dataset"

        validation_dict = get_dataset_metrics(self.validation_set, ignore_list=self.ignore_list,
                                              dataset_value_parser_fn=self.dataset_value_parser)
        comparisons = ""

        for key in list(validation_dict.keys()):
            comparisons += f"![image]({create_bar_plot(data_set=validation_dict[key], feature_name=key, report_location=self.write_directory, normalize=False)})\n"

        return heading + validation_desc + comparisons

    def make_examples_section(self):
        heading = "# Dataset Examples\n"

        section_desc = "This section depicts one input for each label the model is expected to learn.\n"

        examples = create_example_rows(self.train_set, input_key='input', label_key='label',
                                       report_location=self.write_directory)

        return heading + section_desc + examples

    def make_hyperparameters_section(self):
        heading = "# Hyperparameters \n"

        section_desc = "This section documents the hyperparameters used for this session. \n"

        parameters = ""

        for idx, param in enumerate(self.hyper_parameters):
            parameters += f"{idx + 1}. {param}: {self.hyper_parameters[param]}\n"

        return heading + section_desc + parameters

    def make_metrics_section(self):
        metrics_str = "# Performance\n"
        # blurb for section details
        for metric in self.metrics:
            metrics_str += f"![image]({metric.create_plot(self.write_directory)})\n"

        return metrics_str

    def make_matrix_section(self):

        confusion_heading_str = "# Confusion Matrix Against Validation Set\n"

        confusion_desc = "The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. \n"

        confusion_matrix_image = f"![image]({create_image_from_matrix(self.confusion_matrix, self.write_directory)})\n"

        score_heading = "# Score Matrix \n"

        score_matrix = eu.create_measure_matrix(self.confusion_matrix)

        score_matrix_desc = "The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. \n"

        score_matrix_image = f"![image]({create_image_from_matrix(score_matrix, self.write_directory)})\n"

        return confusion_heading_str + confusion_desc + confusion_matrix_image + score_heading + score_matrix_desc + score_matrix_image

    def make_roc_section(self):

        heading = "# Multiclass Receiver Operating Characteristic (ROC) Curves \n"

        desc = "The multiclass ROC curves were created using one-versus-rest classifications against the validation set.\n"

        roc_images_str = ""

        for key in self.roc_dict.keys():
            tpr = self.roc_dict[key]['tp_rates']
            fpr = self.roc_dict[key]['fp_rates']
            label_name = self.label_map_fn(key)
            roc_images_str += f"![image]({create_roc_plot(tpr, fpr, label_name, self.write_directory)})\n"

        return heading + desc + roc_images_str

    def write_report(self, name):
        with open(f"{self.write_directory}/{name}.md", "w") as report_out:
            report_out.write(self.make_introduction_section())
            report_out.write(self.make_hyperparameters_section())
            report_out.write(self.make_metrics_section())
            report_out.write(self.make_datasets_section())
            report_out.write(self.make_examples_section())

    def write_evaluation_report(self, name):
        with open(f"{self.write_directory}/{name}.md", "w") as report_out:
            report_out.write(self.make_matrix_section())
            report_out.write(self.make_roc_section())
            report_out.write(self.make_evaluation_dataset_section())


def cli_main(flags):
    file_list = []
    # find the files in the report directory and any sub directories

    print(f"Cleaning {flags.report_dir}")

    for subdir, dirs, files in os.walk(flags.report_dir):
        for file in files:
            file_list.append(os.path.join(subdir, file))

    # read in reports to one string.
    report_strings = ""
    for file in file_list:
        if file[-3:] == ".md":
            with open(file, 'r') as file_in:
                report_strings += file_in.read()

    # for files not listed in the report, remove them
    for file in file_list:
        if file[-4:] == ".png":
            # get file name
            file_name = file.split("/")[-1]
            if report_strings.find(file_name) == -1:
                print(f"Removing {file}")
                os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_dir', type=str,
                        default="./reports/",
                        help='the location of the reports directory to clean')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
