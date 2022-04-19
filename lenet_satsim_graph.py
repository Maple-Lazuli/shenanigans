import tensorflow as tf
import numpy as np
import argparse
from dataset_generator import DatasetGenerator
from make_report import Report
from metrics import Metric
import json
from datetime import datetime

tf.compat.v1.disable_eager_execution()


def get_parameters(file_name):
    with open(file_name) as json_file:
        parameters = json.load(json_file)
    return parameters


def calculate_mse(prediction, truth):
    """
    Calculates mean squared error using the error from the predictions and truths.
    Parameters
    ----------
    prediction: a list of predictions for each class such that the shape is (, 5)
    truth: a list of truths for each class such that the shape is (, 5)

    Returns
    -------
    MSE score from the errors
    """

    error_sum = 0
    n = 0
    for (p, t) in zip(prediction, truth):
        squared_error = (np.array(p) - np.array(t)) ** 2
        error_sum += np.sum(squared_error)
        n += len(p)

    mean_squared_error = error_sum / n

    # mse = np.mean(np.square(np.sum(np.array(prediction) - np.array(truth))))

    return mean_squared_error


def calculate_mae(prediction, truth):
    """
    Calculates mean absolute error using the error from the predictions and truths.
    Parameters
    ----------
    prediction: a list of predictions for each class such that the shape is (, 5)
    truth: a list of truths for each class such that the shape is (, 5)

    Returns
    -------
    MAE score from the errors
    """

    absolute_error = 0
    n = 0
    for (p, t) in zip(prediction, truth):
        absolute_error += np.sum(np.abs(np.array(p) - np.array(t)))
        n += len(p)

    mean_absolute_error = absolute_error / n

    return mean_absolute_error


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
    if key == "class_name":
        return value.decode('UTF-8')
    elif key == "label":
        return value.argmax()
    else:
        return value


def parse_satsim_record(example_proto):
    """
    A function that describes how to parse the records in the TF Dataset for the DatasetGenerator Class

    Parameters
    ----------
    example_proto: The extracted example from the TF dataset

    Returns
    -------
    A feature dictionary with the proper datatypes for use in the model.
    """
    # Define how to parse the example
    features = {
        "height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "depth": tf.io.FixedLenFeature([], dtype=tf.int64),
        "field_of_view_x": tf.io.FixedLenFeature([], dtype=tf.float32),
        "field_of_view_y": tf.io.FixedLenFeature([], dtype=tf.float32),
        "stray_light": tf.io.FixedLenFeature([], dtype=tf.int64),
        "class_name": tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    # Parse the example
    features_parsed = tf.io.parse_single_example(
        serialized=example_proto, features=features
    )
    width = tf.cast(features_parsed["width"], tf.int64)
    height = tf.cast(features_parsed["height"], tf.int64)
    depth = tf.cast(features_parsed["depth"], tf.int64)
    iFOVx = tf.cast(features_parsed["field_of_view_x"], tf.float32)
    iFOVy = tf.cast(features_parsed["field_of_view_y"], tf.float32)
    stray_light = tf.cast(features_parsed["stray_light"], tf.int64)
    class_name = tf.cast(features_parsed["class_name"], tf.string)
    image = tf.io.decode_raw(features_parsed["image_raw"], tf.uint16)
    image = tf.cast(image, tf.float32)

    label = tf.io.decode_raw(features_parsed["label"], tf.int64)
    label = tf.cast(label, tf.float32)

    return_features = {
        "height": height,
        "width": width,
        "depth": depth,  # 16 bits per pixel for FITS images
        "field_of_view_x": iFOVx,
        "field_of_view_y": iFOVy,
        "stray_light": stray_light,
        "class_name": class_name,
        "label": label,
        "input": image
    }

    return return_features


def get_predictions_from_logits(batch_logits):
    batch_prediction = list()
    for logits in batch_logits:
        element_prediction = np.zeros(len(logits))
        element_prediction[np.argmax(logits)] = 1
        batch_prediction.append(element_prediction)

    return batch_prediction


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


class SatSimModel(object):
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
        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        """
        self.learning_rate = learning_rate
        self.sess = sess
        self.writer = writer

        self.train_iterator = train_dataset.get_iterator()
        self.valid_iterator = valid_dataset.get_iterator()

        self.next_train = self.train_iterator.get_next()
        self.next_valid = self.valid_iterator.get_next()

        with tf.name_scope("lenet_satsim_model"):
            self._build_mnist_model()

        if reporter is not None:
            self.reporter = reporter
            self.reporter.set_introduction(self.desc)
            self.reporter.add_hyperparameter({'learning_rate': self.learning_rate})
            self.reporter.set_dataset_value_parser(dataset_value_parser)

    def train(self, epochs, save=False, save_location=None):

        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()
        self.sess.run(init)

        # Register metrics to go to the reporter
        loss_metric = Metric(title="Loss", horizontal_label="Training Steps", vertical_label="Loss")
        epoch_mse_metric = Metric(title="Mean Squared Error", horizontal_label="Epochs", vertical_label="Error")
        epoch_mae_metric = Metric(title="Mean Absolute Error", horizontal_label="Epochs", vertical_label="Error")

        for i in range(epochs):
            print(f"Started Training Epoch {i}  at {str(datetime.now())}")
            prediction_list = list()
            truth_list = list()
            try:
                self.sess.run(self.train_iterator.initializer)

                while True:
                    features = self.sess.run(self.next_train)
                    batch_x = features['input']
                    batch_y = features['label']
                    (loss, _, batch_logits) = self.sess.run([self.cross_entropy, self.minimize,
                                                             self.predicted_label],
                                                            feed_dict={self.input_image_batch: batch_x,
                                                                       self.true_label: batch_y,
                                                                       self.hold_prob: 0.5})

                    batch_prediction = get_predictions_from_logits(batch_logits)

                    for (y, prediction) in zip(batch_y, batch_prediction):
                        truth_list.append(y)
                        prediction_list.append(prediction)

                    loss_metric.add("loss", loss)

            except tf.errors.OutOfRangeError:
                epoch_mse_metric.add("training_mse", calculate_mse(prediction_list, truth_list))
                epoch_mae_metric.add("training_mae", calculate_mae(prediction_list, truth_list))

                prediction_list = list()
                truth_list = list()
                try:
                    self.sess.run(self.valid_iterator.initializer)
                    while True:
                        features = self.sess.run(self.next_valid)
                        valid_x = features['input']
                        valid_y = features['label']

                        batch_logits = self.inference(valid_x)

                        batch_prediction = get_predictions_from_logits(batch_logits)

                        for (y, prediction) in zip(valid_y, batch_prediction):
                            truth_list.append(y)
                            prediction_list.append(prediction)

                except tf.errors.OutOfRangeError:
                    epoch_mse_metric.add("valid_mse", calculate_mse(prediction_list, truth_list))
                    epoch_mae_metric.add("valid_mae", calculate_mae(prediction_list, truth_list))

        if save:
            saver.save(self.sess, save_location)

        if self.reporter is not None:
            self.reporter.add_hyperparameter({'epochs': epochs})
            self.reporter.add_metric(loss_metric)
            self.reporter.add_metric(epoch_mse_metric)
            self.reporter.add_metric(epoch_mae_metric)

    def _build_mnist_model(self):

        self.input_image_batch = tf.compat.v1.placeholder(tf.float32, shape=[None, 262144], name="X")

        self.true_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 5], name="y_true")

        x_image = tf.reshape(self.input_image_batch, [-1, 512, 512, 1])

        with tf.name_scope("Convolutional_1"):
            convo_1 = convolutional_layer(x_image, shape=[8, 8, 1, 32])
            convo_1_pooling = max_pool_2by2(convo_1)

        with tf.name_scope("Convolutional_2"):
            convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
            convo_2_pooling = max_pool_2by2(convo_2)
            convo_2_flat = tf.reshape(convo_2_pooling, [-1, 128 * 128 * 64])

        with tf.name_scope("Fully_Connected_1"):
            full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        self.hold_prob = tf.compat.v1.placeholder(tf.float32, name="hold_prob")

        with tf.name_scope("Fully_Connected_With_Dropout"):
            full_one_dropout = tf.nn.dropout(full_layer_one, rate=1 - self.hold_prob)

        with tf.name_scope("Y_Prediction"):
            self.predicted_label = normal_full_layer(full_one_dropout, 5)

        with tf.name_scope("Loss"):
            self.cross_entropy = tf.compat.v1.reduce_mean(
                input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits(
                    labels=tf.compat.v1.stop_gradient(self.true_label),
                    logits=self.predicted_label))

        with tf.name_scope("Optimizer"):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        with tf.name_scope("train"):
            self.minimize = optimizer.minimize(self.cross_entropy)

        with tf.name_scope("metrics"):
            # find correct predictions
            matches = tf.compat.v1.equal(tf.compat.v1.argmax(input=self.predicted_label, axis=1),
                                         tf.compat.v1.argmax(input=self.true_label, axis=1))
            # find the number of correct predictions
            self.batch_accuracy = tf.compat.v1.reduce_sum(input_tensor=tf.compat.v1.cast(matches, tf.float32))

    def inference(self, input_image_batch):
        """
        Returns a vector of logits

        Parameters
        ----------
        input_image - a matrix of shape (x, 512,512)

        Returns
        -------
        list
        """
        softmax = tf.compat.v1.math.softmax(self.predicted_label, name="PredictionSoftmax")
        return self.sess.run(softmax, feed_dict={self.input_image_batch: input_image_batch,
                                                 self.hold_prob: 1.0})


def cli_main(flags):
    train_records = flags.train_set_location
    validation_records = flags.validation_set_location

    train_df = DatasetGenerator(train_records, parse_function=parse_satsim_record, shuffle=True,
                                batch_size=flags.train_batchsize)
    validation_df = DatasetGenerator(validation_records, parse_function=parse_satsim_record, shuffle=True,
                                     batch_size=flags.validate_batchsize)

    reporter = Report()
    reporter.set_train_set(train_df)
    reporter.set_validation_set(validation_df)
    reporter.set_write_directory(flags.report_dir)
    reporter.set_ignore_list([
        "input",
        "depth",
        "width",
        "height",
        "field_of_view_x",
        "field_of_view_y"
    ])
    with tf.compat.v1.Session() as sess:
        model = SatSimModel(sess, train_df, validation_df, learning_rate=flags.learning_rate, reporter=reporter)
        model.train(epochs=flags.epochs, save=flags.save, save_location=flags.model_save_dir)

    reporter.write_report(f"{flags.report_name_base}_train_{str(datetime.now())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='The number of epochs for training')

    parser.add_argument('--learning_rate', type=float,
                        default=0.001,
                        help='The learning rate to use during training')

    parser.add_argument('--train_set_location', type=str,
                        default="/media/ada/Internal Expansion/shenanigans_storage/generated_data_df/train/satsim_train.tfrecords",
                        help='The location of the training set')

    parser.add_argument('--validation_set_location', type=str,
                        default="/media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords",
                        help='The location of the validation set')

    parser.add_argument('--train_batchsize', type=int,
                        default=50,
                        help='The batch size to use for feeding training examples')

    parser.add_argument('--validate_batchsize', type=int,
                        default=50,
                        help='The batch size to use for feeding validation examples')

    parser.add_argument('--model_save_dir', type=str,
                        default="/media/ada/Internal Expansion/shenanigans_storage/lenet_satsim_model/satsim",
                        help='The directory to save the model in.')

    parser.add_argument('--report_name_base', type=str,
                        default="lenet-satsim",
                        help='The base name for the report')

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
