import tensorflow as tf
import numpy as np
import evaluation_utils as eu
from dataset_generator import DatasetGenerator
import argparse
import json
import lenet_mnist_graph as mnist
import make_tfrecords_mnist as mnist_records
import lenet_satsim_graph as satsim
import make_tfrecords_satsim as satsim_records
from datetime import datetime
from make_report import Report


def get_parameters(file_name):
    with open(file_name) as json_file:
        parameters = json.load(json_file)
    return parameters


def labels_from_classifications(classifications):
    labels = set()
    for classification in classifications:
        labels.add(classification[-1])
    return list(labels)


def cli_main(flags):
    reporter = Report()
    sess = tf.compat.v1.Session()
    # read in the configuration json
    config_dict = get_parameters(flags.config_json)

    # restore model with configuration json
    saver = tf.compat.v1.train.import_meta_graph(config_dict['graph_location'])
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(config_dict['checkpoint_dir']))
    graph = tf.compat.v1.get_default_graph()
    input_image = graph.get_tensor_by_name(config_dict['input_tensor_name'])
    classifier_label = graph.get_tensor_by_name(config_dict['classifier_tensor_name'])
    softmax_classifier = tf.compat.v1.math.softmax(classifier_label)
    hold_prob = graph.get_tensor_by_name(config_dict['hold_prob_name'])

    if config_dict['graph_name'] == "mnist":
        parse_fn = mnist.parse_records
        reporter.set_dataset_value_parser(mnist.dataset_value_parser)
        reporter.set_label_map_fn(mnist_records.map_label_to_name)
    elif config_dict['graph_name'] == 'satsim':
        parse_fn = satsim.parse_satsim_record
        reporter.set_dataset_value_parser(satsim.dataset_value_parser)
        reporter.set_label_map_fn(satsim_records.map_label_to_name)

    valid_df = DatasetGenerator(config_dict['validation_set_location'], parse_function=parse_fn, shuffle=True,
                                batch_size=1)
    iterator = valid_df.get_iterator()
    next_step = iterator.get_next()

    reporter.set_validation_set(valid_df)

    reporter.set_ignore_list(config_dict['ignore_list'])

    classifications = None
    try:
        sess.run(iterator.initializer)
        while True:
            features = sess.run(next_step)
            batch_x = features['input']
            batch_y = features['label']
            y_pred = sess.run(softmax_classifier, feed_dict={input_image: batch_x, hold_prob: 1.0})

            # find the true class from the batch
            true_class = np.argmax(batch_y)

            # make an np array of the predictions and true class
            classifications_and_true_class = np.hstack([y_pred[0], true_class])

            # add the np array from the previous step to a growing matrix
            classifications = np.vstack([classifications,
                                         classifications_and_true_class]) if classifications is not None else classifications_and_true_class

    except tf.errors.OutOfRangeError:
        print("Finished Classifications against the dataset")

    # Create a list of labels seen during classification
    labels = labels_from_classifications(classifications)

    # Create a confusion matrix with the labels seen during classification
    confusion_matrix = eu.create_confusion_matrix(classifications, labels)
    print("Finished Creating Confusion Matrix")

    # Create a dictionary for ROC Curves
    roc_dict = eu.create_ovr_roc_dict(classifications, len(labels), 1000)
    print("Finished ROC Calculations")
    reporter.set_confusion_matrix(confusion_matrix)
    reporter.set_roc_dict(roc_dict)
    reporter.set_write_directory(flags.report_dir)
    reporter.write_evaluation_report(f"{config_dict['report_name_base']}_evaluate_{str(datetime.now())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_json', type=str,
                        default='./mnist_config.json',
                        help="The config json file containing the parameters for the model")

    parser.add_argument('--report_dir', type=str,
                        default='./reports/',
                        help='Where to save the reports.')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
