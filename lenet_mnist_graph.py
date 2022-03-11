import tensorflow as tf
import numpy as np
import argparse
from dataset_generator import DatasetGenerator

tf.compat.v1.disable_eager_execution()


def parse_records(example_proto):
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
    image = tf.io.decode_raw(features_parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.io.decode_raw(features_parsed["label"], tf.int64)
    label = tf.cast(label, tf.float32)

    return image, label


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
                 writer=None):
        self.learning_rate = learning_rate
        self.sess = sess
        self.writer = writer

        self.train_iterator = train_dataset.get_iterator()
        self.valid_iterator = valid_dataset.get_iterator()

        self.next_train = self.train_iterator.get_next()
        self.next_valid = self.valid_iterator.get_next()

        with tf.name_scope("mnist_model"):
            self.minimize, self.X, self.y_pred, self.y_true, self.hold_prob = self._build_mnist_model()

    def train(self, epochs, save_location=None):
        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()
        self.sess.run(init)
        for i in range(epochs):
            try:
                self.sess.run(self.train_iterator.initializer)
                while True:
                    batch_x, batch_y = self.sess.run(self.next_train)
                    self.sess.run(self.minimize,
                                  feed_dict={self.X: batch_x,
                                             self.y_true: batch_y,
                                             self.hold_prob: 0.5})
            except tf.errors.OutOfRangeError:
                if i % 2 == 0:
                    try:
                        # TODO refactor self.train_iterator.get_next() out of the training loop
                        self.sess.run(self.valid_iterator.initializer)
                        running_sum = 0
                        number_of_comparisons = 0
                        while True:
                            test_x, test_y = self.sess.run(self.next_valid)
                            # find correct predictions
                            matches = tf.compat.v1.equal(tf.compat.v1.argmax(input=self.y_pred, axis=1),
                                                         tf.compat.v1.argmax(input=self.y_true, axis=1))
                            # find the number of correct predictions
                            acc = tf.compat.v1.reduce_sum(input_tensor=tf.compat.v1.cast(matches, tf.float32))
                            # add the number of correct predictions to the running sum
                            running_sum += self.sess.run(acc, feed_dict={self.X: test_x, self.y_true: test_y,
                                                                         self.hold_prob: 1.0})
                            number_of_comparisons += test_y.shape[0]

                    except tf.errors.OutOfRangeError:
                        print(
                            f"accuracy for epoch {i} is: {running_sum / number_of_comparisons} "
                            f"from successes  {running_sum} out of {number_of_comparisons} trials")
        if save_location is not None:
            saver.save(self.sess, save_location)

    def _build_mnist_model(self):

        X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="X")

        y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name="y_true")

        x_image = tf.reshape(X, [-1, 28, 28, 1])

        with tf.name_scope("Convolutional_1"):
            convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
            convo_1_pooling = max_pool_2by2(convo_1)

        with tf.name_scope("Convolutional_2"):
            convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
            convo_2_pooling = max_pool_2by2(convo_2)
            convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])

        with tf.name_scope("Fully_Connected_1"):
            full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        hold_prob = tf.compat.v1.placeholder(tf.float32, name="hold_prob")
        with tf.name_scope("Fully_Connected_With_Dropout"):
            full_one_dropout = tf.nn.dropout(full_layer_one, rate=1 - hold_prob)

        with tf.name_scope("Y_Prediction"):
            y_pred = normal_full_layer(full_one_dropout, 10)

        with tf.name_scope("Loss"):
            cross_entropy = tf.compat.v1.reduce_mean(
                input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits(
                    labels=tf.compat.v1.stop_gradient(y_true),
                    logits=y_pred))

        with tf.name_scope("Optimizer"):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        with tf.name_scope("train"):
            minimize = optimizer.minimize(cross_entropy)

        return minimize, X, y_pred, y_true, hold_prob


def cli_main(flags):
    train_records = "./mnist_tf/train/mnist_train.tfrecords"
    test_records = "./mnist_tf/test/mnist_test.tfrecords"

    train_df = DatasetGenerator(train_records, parse_function=parse_records, shuffle=True, batch_size=flags.batch_size)
    test_df = DatasetGenerator(test_records, parse_function=parse_records, shuffle=True, batch_size=flags.batch_size)

    with tf.compat.v1.Session() as sess:
        sess.run
        model = MNISTModel(sess, train_df, test_df, learning_rate=flags.learning_rate)
        model.train(epochs=flags.epochs, save_location=flags.save_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='The number of epochs for training')

    parser.add_argument('--learning_rate', type=int,
                        default=0.001,
                        help='The learning rate to use during training')
    parser.add_argument('--save_location', type=str,
                        default="./model/model.ckpt",
                        help='The location to save the model in')

    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='the number of images to train on at once')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
