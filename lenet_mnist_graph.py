import tensorflow as tf
import numpy as np
import argparse
from dataset_generator import DatasetGenerator
from make_report import Report
from metrics import Metric
import json

tf.compat.v1.disable_eager_execution()


def get_parameters(file_name):
    with open(file_name) as json_file:
        parameters = json.load(json_file)
    return parameters


def calculate_mse(scores):
    """
    Calculates mean squared error using the error from the training steps.
    Parameters
    ----------
    scores: A list of errors

    Returns
    -------
    MSE score from the list of errors
    """
    scores = np.array(scores)
    return np.sum((50 - scores) * (50 - scores)) / len(scores)


def calculate_mae(scores):
    """
    Calculates mean absolute error using the error from the training steps.
    Parameters
    ----------
    scores: A list of errors

    Returns
    -------
    MAE score from the list of errors
    """
    scores = np.array(scores)
    return np.sum((50 - scores)) / len(scores)


def dataset_value_parser(key, value):
    """
    A function to be passed to the reporter instance that performs transformations to some features to help with
    rendering in a report.

    Parameters
    ----------
    key: The key of the feature.
    value: The value of the feature

    Returns
    -------
    Returns either the feature or a transformation of the feature
    """
    if key == "label":
        return value.argmax()
    else:
        return value


def parse_records(example_proto):
    """
    A function that describes how to parse the records in the TF Dataset for the DatasetGenerator Class

    Parameters
    ----------
    example_proto: The extracted example from the TF dataset

    Returns
    -------
    A feature dictionary with the proper datatypes for use in the model.
    """
    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    features_parsed = tf.io.parse_single_example(
        serialized=example_proto, features=features
    )

    width = tf.cast(features_parsed["width"], tf.int64)
    height = tf.cast(features_parsed["height"], tf.int64)
    depth = tf.cast(features_parsed["depth"], tf.int64)

    image = tf.io.decode_raw(features_parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.io.decode_raw(features_parsed["label"], tf.int64)
    label = tf.cast(label, tf.float32)

    return_features = {
        "height": height,
        "width": width,
        "depth": depth,
        "label": label,
        "input": image
    }

    return return_features


def init_weights(shape):
    init_random_dist = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.compat.v1.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.compat.v1.constant(0.1, shape=shape)
    return tf.compat.v1.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.compat.v1.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.compat.v1.nn.max_pool2d(input=x, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.compat.v1.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.compat.v1.add(tf.compat.v1.matmul(input_layer, W), b, name="y_pred")


class MNISTModel(object):
    def __init__(self,
                 sess,
                 train_dataset,
                 valid_dataset,
                 inputs=None,
                 learning_rate=1.0,
                 writer=None,
                 reporter=None):

        # Used in the overview section of the reporter
        self.desc = """
        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        """

        self.reporter = reporter
        self.learning_rate = learning_rate
        self.sess = sess
        self.writer = writer

        self.train_iterator = train_dataset.get_iterator()
        self.valid_iterator = valid_dataset.get_iterator()

        self.next_train = self.train_iterator.get_next()
        self.next_valid = self.valid_iterator.get_next()

        with tf.name_scope("mnist_model"):
            self._build_mnist_model()

        if reporter is not None:
            self.reporter = reporter
            self.reporter.set_introduction(self.desc)
            self.reporter.add_hyperparameter({'learning_rate': self.learning_rate})
            self.reporter.set_dataset_value_parser(dataset_value_parser)

    def train(self, epochs, save, save_location=None):
        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()
        self.sess.run(init)

        # Register metrics to go to the reporter
        loss_metric = Metric(title="Loss", horizontal_label="Training Steps", vertical_label="Loss")
        epoch_mse_metric = Metric(title="Mean Squared Error", horizontal_label="Epochs", vertical_label="Error")
        epoch_mae_metric = Metric(title="Mean Absolute Error", horizontal_label="Epochs", vertical_label="Error")

        for i in range(epochs):
            try:
                self.sess.run(self.train_iterator.initializer)
                batch_match_list = []
                while True:
                    features = self.sess.run(self.next_train)
                    batch_x = features['input']
                    batch_y = features['label']

                    (loss, _, batch_accuracy) = self.sess.run([self.cross_entropy, self.minimize,
                                                               self.batch_accuracy],
                                                              feed_dict={self.image_input: batch_x,
                                                                         self.actual_label: batch_y,
                                                                         self.hold_prob: 0.5})

                    loss_metric.add("loss", loss)
                    batch_match_list.append(batch_accuracy)

            except tf.errors.OutOfRangeError:
                epoch_mse_metric.add("training_loss", calculate_mse(batch_match_list))
                epoch_mae_metric.add("training_loss", calculate_mae(batch_match_list))
                try:
                    self.sess.run(self.valid_iterator.initializer)
                    batch_match_list = []
                    while True:
                        features = self.sess.run(self.next_valid)
                        validate_x = features['input']
                        validate_y = features['label']
                        # Find occurrences where there are matches
                        matches = tf.compat.v1.equal(tf.compat.v1.argmax(input=self.predicted_label, axis=1),
                                                     tf.compat.v1.argmax(input=self.actual_label, axis=1))
                        # Find the number of correct predictions
                        acc = tf.compat.v1.reduce_sum(input_tensor=tf.compat.v1.cast(matches, tf.float32))
                        # add the number of correct predictions to the running sum
                        score = self.sess.run(acc, feed_dict={self.image_input: validate_x,
                                                              self.actual_label: validate_y,
                                                              self.hold_prob: 1.0})
                        batch_match_list.append(score)
                except tf.errors.OutOfRangeError:
                    epoch_mse_metric.add("validation_loss", calculate_mse(batch_match_list))
                    epoch_mae_metric.add("validation_loss", calculate_mae(batch_match_list))
        if save:
            saver.save(self.sess, save_location)

        if self.reporter is not None:
            self.reporter.add_hyperparameter({'epochs': epochs})
            self.reporter.add_metric(loss_metric)
            self.reporter.add_metric(epoch_mse_metric)
            self.reporter.add_metric(epoch_mae_metric)

    def _build_mnist_model(self):

        self.image_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="X")
        self.actual_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name="y_true")

        x_image = tf.reshape(self.image_input, [-1, 28, 28, 1])

        with tf.name_scope("Convolutional_1"):
            convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
            convo_1_pooling = max_pool_2by2(convo_1)

        with tf.name_scope("Convolutional_2"):
            convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
            convo_2_pooling = max_pool_2by2(convo_2)
            convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])

        with tf.name_scope("Fully_Connected_1"):
            full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        self.hold_prob = tf.compat.v1.placeholder(tf.float32, name="hold_prob")
        with tf.name_scope("Fully_Connected_With_Dropout"):
            full_one_dropout = tf.nn.dropout(full_layer_one, rate=1 - self.hold_prob)

        with tf.name_scope("Y_Prediction"):
            self.predicted_label = normal_full_layer(full_one_dropout, 10)

        with tf.name_scope("Loss"):
            self.cross_entropy = tf.compat.v1.reduce_mean(
                input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits(
                    labels=tf.compat.v1.stop_gradient(self.actual_label),
                    logits=self.predicted_label))

        with tf.name_scope("Optimizer"):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        with tf.name_scope("train"):
            self.minimize = optimizer.minimize(self.cross_entropy)

        with tf.name_scope("metrics"):
            # find correct predictions
            matches = tf.compat.v1.equal(tf.compat.v1.argmax(input=self.predicted_label, axis=1),
                                         tf.compat.v1.argmax(input=self.actual_label, axis=1))
            # find the number of correct predictions
            self.batch_accuracy = tf.compat.v1.reduce_sum(input_tensor=tf.compat.v1.cast(matches, tf.float32))


def cli_main(flags):
    config_dict = get_parameters(flags.config_json)

    train_records = config_dict['train_set_location']
    validation_records = config_dict['validation_set_location']

    train_df = DatasetGenerator(train_records, parse_function=parse_satsim_record, shuffle=True,
                                batch_size=flags.batch_size)
    validation_df = DatasetGenerator(validation_records, parse_function=parse_satsim_record, shuffle=True,
                                     batch_size=flags.batch_size)

    reporter = Report()
    reporter.set_train_set(train_df)
    reporter.set_validation_set(validation_df)
    reporter.set_write_directory(flags.report_dir)
    reporter.set_ignore_list(config_dict['idgnore_list'])

    with tf.compat.v1.Session() as sess:
        # sess.run # i dont remember why this is here
        model = MNISTModel(sess, train_df, validation_df, learning_rate=flags.learning_rate, reporter=reporter)
        model.train(epochs=flags.epochs, save=flags.save, save_location=flags.save_location)

    reporter.write_report(flags.report_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='The number of epochs for training')

    parser.add_argument('--learning_rate', type=int,
                        default=0.001,
                        help='The learning rate to use during training')

    parser.add_argument('--config_json', type=str,
                        default='./mnist_config.json',
                        help="The config json file containing the parameters for the model")

    parser.add_argument('--save', type=bool,
                        default=True,
                        help='Whether or not to save the model')

    parser.add_argument('--report', type=bool,
                        default=True,
                        help='Whether to create a report.')

    parser.add_argument('--report_dir', type=str,
                        default='./reports/',
                        help='Where to save the reports.')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
