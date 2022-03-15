import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import hashlib
from collections import Counter

"""
Doc string here
"""


def get_dataset_report_examples(dataset, label_key):
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
    examples = get_dataset_report_examples(dataset, label_key)
    return_str = ""

    for idx,example in enumerate(examples):
        image = example[input_key].reshape(example['height'], example['width'])
        return_str += f"### Example {idx + 1}"
        for idx, key in enumerate(example.keys()):
            if key != input_key:
                return_str += f"{idx}. {key}:{example[key]}\n"
        return_str += f"![image]({create_raster_image(image, report_location)})\n"
    return return_str


def create_raster_image(data, report_location):
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
    return value


def normalize_dict(d):
    summ = 0
    for key in d.keys():
        summ += d[key]
    for key in d.keys():
        d[key] = d[key] / summ
    return d


def create_bar_plot(train_set, test_set, feature_name, report_location, normalize=False):
    if normalize:
        train_counter = normalize_dict(Counter(train_set))
        test_counter = normalize_dict(Counter(test_set))
    else:
        train_counter = Counter(train_set)
        test_counter = Counter(test_set)

    keys = set()
    [keys.add(key) for key in train_counter.keys()]
    [keys.add(key) for key in test_counter.keys()]

    N = len(keys)

    train_values = []
    test_values = []

    for key in keys:
        if key not in train_counter.keys():
            train_values.append(0)
        else:
            train_values.append(train_counter[key])

        if key not in test_counter.keys():
            test_values.append(0)
        else:
            test_values.append(test_counter[key])

    # Position of bars on x-axis
    ind = np.arange(N)
    width = 0.3
    # Figure size

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(ind, train_values, width, label='Train Set')
    ax.bar(ind + width, test_values, width, label='Test Set')
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
        """

        """
        self.introduction = None
        self.metrics = []
        self.test_set = None
        self.train_set = None
        self.write_directory = None
        self.hyper_parameters = dict()
        self.dataset_value_parser = dataset_value_parser
        self.ignore_list = []

    # put setters here

    def set_introduction(self, introduction):
        self.introduction = introduction

    def add_metric(self, metrics):
        self.metrics.append(metrics)

    def add_hyperparameter(self, param):
        self.hyper_parameters.update(param)

    def set_test_set(self, test_set):
        self.test_set = test_set

    def set_train_set(self, train_set):
        self.train_set = train_set

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

    def make_dataset_section(self):
        heading = "# Datasets \n"

        training_heading = "### Training Set \n"

        training_desc = f"The training set located at {self.train_set.get_location()} consists of {self.train_set.get_size()}, served in batch sizes of {self.train_set.get_batch_size()}.\n"

        # create a bar chart comparing feature counts.

        test_heading = "### Testing Set \n"

        test_desc = f"The testing set located at {self.test_set.get_location()} consists of {self.test_set.get_size()}, served in batch sizes of {self.test_set.get_batch_size()}.\n"

        comparison_heading = "### Test Set / Training Set Comparison \n"

        comparison_desc = "This section compares the contents of the test and train sets used.\n"

        train_dict = get_dataset_metrics(self.train_set, ignore_list=self.ignore_list,
                                         dataset_value_parser_fn=self.dataset_value_parser)
        test_dict = get_dataset_metrics(self.test_set, ignore_list=self.ignore_list,
                                        dataset_value_parser_fn=self.dataset_value_parser)
        comparisons = ""
        for key in list(train_dict.keys()) + list(test_dict.keys()):
            comparisons += f"![image]({create_bar_plot(train_dict[key], test_dict[key], key, self.write_directory)})\n"
            comparisons += f"![image]({create_bar_plot(train_dict[key], test_dict[key], key, self.write_directory, normalize=True)})\n "

        return heading + training_heading + training_desc + test_heading + test_desc + comparison_heading + comparison_desc + comparisons

    def make_examples_section(self):

        heading = "# Dataset Examples\n"

        section_desc = "This section depicts one input for each label the model is expected to learn.\n"

        examples = create_example_rows(self.train_set, input_key = 'input', label_key = 'label', report_location = self.write_directory)

        return heading + section_desc + examples

    def make_hyperparameters_section(self):
        heading = "# Hyperparameters \n"

        section_desc = "This section documents the hyperparameters used for this session. \n"

        parameters = ""

        for idx, param in enumerate(self.hyper_parameters):
            parameters += f"{idx + 1}. {param}: {self.hyper_parameters[param]}\n"

        return heading + section_desc + parameters

    def make_metrics_section(self):
        metrics_str = "# Metrics\n"
        # blurb for section details
        for metric in self.metrics:
            metrics_str += f"![image]({metric.create_plot(self.write_directory)})\n"

        return metrics_str

    def write_report(self, name):
        with open(f"{self.write_directory}/{name}.md", "w") as report_out:
            report_out.write(self.make_introduction_section())
            report_out.write(self.make_hyperparameters_section())
            report_out.write(self.make_metrics_section())
            report_out.write(self.make_dataset_section())
            report_out.write(self.make_examples_section())

