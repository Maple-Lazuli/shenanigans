import tensorflow as tf
import numpy as np
import evaluation_utils as eu
from dataset_generator import DatasetGenerator
import argparse
import json
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

    # restore model with configuration json
    saver = tf.compat.v1.train.import_meta_graph(flags.graph_location)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(flags.checkpoint_dir))
    graph = tf.compat.v1.get_default_graph()
    input_image = graph.get_tensor_by_name(flags.input_tensor_name)
    classifier_label = graph.get_tensor_by_name(flags.classifier_tensor_name)
    softmax_classifier = tf.compat.v1.math.softmax(classifier_label)
    hold_prob = graph.get_tensor_by_name(flags.hold_prob_name)

    parse_fn = satsim.parse_satsim_record
    reporter.set_dataset_value_parser(satsim.dataset_value_parser)
    reporter.set_label_map_fn(satsim_records.map_label_to_name)

    valid_df = DatasetGenerator(flags.validation_set_location, parse_function=parse_fn, shuffle=True,
                                batch_size=flags.batch_size)
    iterator = valid_df.get_iterator()
    next_step = iterator.get_next()

    reporter.set_validation_set(valid_df)

    reporter.set_ignore_list([
        "input",
        "depth",
        "width",
        "height",
        "field_of_view_x",
        "field_of_view_y"
    ])

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
    reporter.write_evaluation_report(f"{flags.report_name_base}_evaluate_{str(datetime.now())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph_location', type=str,
                        default='/media/ada/Internal Expansion/shenanigans_storage/lenet_satsim_model/satsim.meta',
                        help='The location of the meta graph')

    parser.add_argument('--checkpoint_dir', type=str,
                        default='/media/ada/Internal Expansion/shenanigans_storage/lenet_satsim_model',
                        help='The location of the model checkpoint')

    parser.add_argument('--input_tensor_name', type=str,
                        default='lenet_satsim_model/X:0',
                        help='The name of the input tensor in the meta graph')

    parser.add_argument('--classifier_tensor_name', type=str,
                        default="lenet_satsim_model/Y_Prediction/y_pred:0",
                        help='The name of the tensor that contains the classification probabilites')

    parser.add_argument('--hold_prob_name', type=str,
                        default="lenet_satsim_model/hold_prob:0",
                        help='The tensor with the hold probabilites')

    parser.add_argument('--validation_set_location', type=str,
                        default="/media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords",
                        help='the location of the validation examples to use for evaluation')

    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='The batch size to use for feeding validation examples')


    parser.add_argument('--report_name_base', type=str,
                        default="lenet_satsim",
                        help='the base name for the reports.')

    parser.add_argument('--report_dir', type=str,
                        default='./reports/',
                        help='Where to save the reports.')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
