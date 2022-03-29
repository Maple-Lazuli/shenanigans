import tensorflow as tf
import numpy as np
from multiclass_confusion_matrix import create_confusion_matrix
from dataset_generator import DatasetGenerator
from lenet_mnist_graph import parse_records
from make_report import Report
from lenet_mnist_graph import dataset_value_parser

if __name__ == '__main__':

    # restore the model
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph("./mnist_model/mnist.meta")
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./mnist_model/"))
    graph = tf.compat.v1.get_default_graph()

    # create references to the tensors
    input_image = graph.get_tensor_by_name("mnist_model/X:0")
    predicted_label = graph.get_tensor_by_name("mnist_model/Y_Prediction/y_pred:0")
    prediction_softmax = tf.compat.v1.math.softmax(predicted_label)
    hold_prob = graph.get_tensor_by_name("mnist_model/hold_prob:0")
    predicted_sigmoid = tf.compat.v1.sigmoid(predicted_label)

    # instantiate the validation dataset iterator
    location = "./mnist_tf/valid/mnist_valid.tfrecords"
    valid_df = DatasetGenerator(location, parse_function=parse_records, shuffle=True,
                               batch_size=1)
    iterator = valid_df.get_iterator()
    next_step = iterator.get_next()

    # create an empty confusion matrix to fill
    confusion_matrix = create_confusion_matrix(list(range(0,10)))

    evaluation = None
    try:
        sess.run(iterator.initializer)
        while True:
            # extract from the dataset
            features = sess.run(next_step)
            batch_x = features['input']
            batch_y = features['label']

            predicted_class = np.argmax(sess.run(predicted_label, feed_dict={input_image: batch_x, hold_prob: 1.0}))
            true_class = np.argmax(batch_y)

            confusion_matrix.iloc[predicted_class, true_class] += 1

    except tf.errors.OutOfRangeError:
        print("Done")

    reporter = Report()
    reporter.set_test_set(valid_df)
    reporter.set_confusion_matrix(confusion_matrix)
    reporter.set_dataset_value_parser(dataset_value_parser)
    reporter.set_ignore_list(['input', 'depth'])
    reporter.set_write_directory('./reports/')
    reporter.write_evaluation_report('lenet_mnist_evaluation')

