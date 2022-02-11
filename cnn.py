"""
Creates The Model
"""
import tensorflow.compat.v1 as tf


def init_weights(shape):
    init_random_dist = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool2d(input=x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


def prepare_model(learning_rate):
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
    convo_1_pooling = max_pool_2by2(convo_1)

    convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
    convo_2_pooling = max_pool_2by2(convo_2)
    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])

    full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))
    hold_prob = tf.compat.v1.placeholder(tf.float32)

    full_one_dropout = tf.nn.dropout(full_layer_one, rate=1 - hold_prob)
    y_pred = normal_full_layer(full_one_dropout, 10)

    cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true), logits=y_pred))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cross_entropy)

    return train, x, y_pred, y_true, hold_prob


class CNN:

    def __init__(self, learning_rate):
        self.train, self.x, self.y_pred, self.y_true, self.hold_prob = prepare_model(learning_rate)
