import tensorflow as tf
import numpy as np
import evaluation_utils as eu
from dataset_generator import DatasetGenerator
from lenet_mnist_graph import parse_records
from make_report import Report
from lenet_mnist_graph import dataset_value_parser

if __name__ == '__main__':
    # instantiate model and dataset
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph("./mnist_model/mnist.meta")
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./mnist_model/"))

    graph = tf.compat.v1.get_default_graph()

    input_image = graph.get_tensor_by_name("mnist_model/X:0")
    predicted_label = graph.get_tensor_by_name("mnist_model/Y_Prediction/y_pred:0")
    prediction_softmax = tf.compat.v1.math.softmax(predicted_label)
    hold_prob = graph.get_tensor_by_name("mnist_model/hold_prob:0")

    # instantiate dataset
    location = "./mnist_tf/valid/mnist_valid.tfrecords"
    valid_df = DatasetGenerator(location, parse_function=parse_records, shuffle=True,
                               batch_size=1)
    iterator = valid_df.get_iterator()
    next_step = iterator.get_next()

    # variable to hold classifications against the validation set
    classifications = None

    try:
        sess.run(iterator.initializer)
        while True:
            features = sess.run(next_step)
            batch_x = features['input']
            batch_y = features['label']
            y_pred = sess.run(prediction_softmax, feed_dict={input_image: batch_x, hold_prob: 1.0})

            # find the true class from the batch
            true_class = np.argmax(batch_y)

            # make an np array of the predictions and true class
            classifications_and_true_class = np.hstack([y_pred[0], true_class])

            # add the np array from the previous step to a growing matrix
            classifications = np.vstack([classifications,
                                         classifications_and_true_class]) if classifications is not None else classifications_and_true_class

    except tf.errors.OutOfRangeError:
        pass

    print("Finished Classifications")
    #to do:refactor to a a dictionary pull from the graph
    labels = list(range(0,10))
    confusion_matrix = eu.create_confusion_matrix(classifications, labels)
    print("Created Confusion Matrix")
    roc_dict = eu.create_ovr_roc_dict(classifications, len(labels), 1000)
    print("Create ROC Dictionary")
    reporter = Report()
    reporter.set_validation_set(valid_df)
    reporter.set_confusion_matrix(confusion_matrix)
    reporter.set_roc_dict(roc_dict)
    reporter.set_dataset_value_parser(dataset_value_parser)
    reporter.set_ignore_list(['input', 'depth'])
    reporter.set_write_directory('./reports/')
    reporter.write_evaluation_report('lenet_mnist_evaluation')

