import tensorflow as tf
import numpy as np
from collections import Counter
from dataset_generator import DatasetGenerator
from lenet_mnist_graph import parse_records
from evalutions import EvaluationMetric
from make_report import Report
from lenet_mnist_graph import dataset_value_parser
def confusion_matrix_calc(actual_class, eval_class, predicted_class, threshold):
    if eval_class == actual_class:
        # The actual class is positive
        if predicted_class > threshold:
            # True Positive
            return "TP"
        else:
            # False Negative
            return "FN"
    else:
        # The actual class is negative
        if predicted_class > threshold:
            # False Positive
            return "FP"
        else:
            # True negative
            return "TN"


def calc_tpr_fpr(counter_dict):
    """
    Calculate the true positive rate and the false positive rate from a counter dictionary
    Parameters
    ----------
    counter_dict: a counter dict containing the confusion matrix labels

    Returns
    -------
    a tuple for the true positive rate and the false positive rate.
    """
    if "TP" not in counter_dict.keys():
        TP = 0
    else:
        TP = counter_dict['TP']

    if "TN" not in counter_dict.keys():
        TN = 0
    else:
        TN = counter_dict['TN']

    if "FP" not in counter_dict.keys():
        FP = 0
    else:
        FP = counter_dict['FP']

    if "FN" not in counter_dict.keys():
        FN = 0
    else:
        FN = counter_dict['FN']

    tpr = float(TP) / float(TP + FN) if TP != 0 else 0
    fpr = float(FP) / float(FP + TN) if FP != 0 else 0

    return tpr, fpr


if __name__ == '__main__':

    # restore the model
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph("./mnist_model/mnist.meta")
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./mnist_model/"))
    graph = tf.compat.v1.get_default_graph()
    # create references to the tensors
    input_image = graph.get_tensor_by_name("mnist_model/X:0")
    predicted_label = graph.get_tensor_by_name("mnist_model/Y_Prediction/y_pred:0")
    hold_prob = graph.get_tensor_by_name("mnist_model/hold_prob:0")
    predicted_sigmoid = tf.compat.v1.sigmoid(predicted_label)

    # instantiate the validation dataset iterator
    location = "./mnist_tf/test/mnist_test.tfrecords"
    test_df = DatasetGenerator(location, parse_function=parse_records, shuffle=True,
                               batch_size=1)
    iterator = test_df.get_iterator()
    next_step = iterator.get_next()

    # Capture performance against the validation dataset

    evaluation = None
    try:
        sess.run(iterator.initializer)
        while True:
            # extract from the dataset
            features = sess.run(next_step)
            batch_x = features['input']
            batch_y = features['label']

            # get the predictions for the labels
            y_pred = sess.run(predicted_sigmoid, feed_dict={input_image: batch_x, hold_prob: 1.0})
            # capture the true label
            actual_class = np.argmax(batch_y)

            # store the predictions against the class
            y_pred_and_class = np.hstack([y_pred[0], actual_class])
            evaluation = np.vstack([evaluation, y_pred_and_class]) if evaluation is not None else y_pred_and_class

    except tf.errors.OutOfRangeError:
        pass

    # Calculate ROC Values
    # a dictionary to hold the tp and fp rates
    roc_dict = {}
    for label in range(0, 10):
        roc_dict[label] = EvaluationMetric(f"ROC For Class {label}", "False Positive Rate", "True Positive Rate")

    for threshold in np.linspace(0.0, 1.0, num=1000):
        # a confusion dictionary to hold the counts from the confusion matrix
        confusion_dict = {}
        for class_pred in range(0, 10):
            confusion_dict[class_pred] = Counter()

        # for each step, evaluate performance against class prediction.
        for step in evaluation:
            actual_class = int(step[-1])

            for class_pred in range(0, len(step) - 1):
                confusion_dict[class_pred].update(
                    [confusion_matrix_calc(actual_class, class_pred, step[class_pred], threshold=threshold)])

        # after evaluating the confusion matrix at a threshold for each step, calculate and store the tpr and fpr
        for key in confusion_dict.keys():
            tpr, fpr = calc_tpr_fpr(confusion_dict[key])
            roc_dict[key].add(v_value=tpr, h_value=fpr)

    reporter = Report()
    reporter.set_test_set(test_df)
    reporter.set_dataset_value_parser(dataset_value_parser)
    reporter.set_ignore_list(['input', 'depth'])
    reporter.set_write_directory('./reports/')
    for key in roc_dict.keys():
        reporter.add_evaluatation_metric(roc_dict[key])
    reporter.write_evaluation_report('lenet_mnist_evaluation')

