import tensorflow.compat.v1 as tf
import argparse

import utils

import cnn

tf.disable_eager_execution()
tf.disable_v2_behavior()

path = "./mnist/"
train_imgs = "train-images.idx3-ubyte"
train_labs = "train-labels.idx1-ubyte"
test_imgs = "t10k-images.idx3-ubyte"
test_labs = "t10k-labels.idx1-ubyte"

train_images = utils.read_mnist_images(path + train_imgs)
test_images = utils.read_mnist_images(path + test_imgs)

train_labels = utils.read_mnist_labels(path + train_labs)
test_labels = utils.read_mnist_labels(path + test_labs)


def train(model, epochs, save_location):
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.train.Saver()
    steps = epochs

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for i in range(steps):
            batch_x, batch_y = train_images[50 * i: 50 * (i + 1)], train_labels[50 * i: 50 * (i + 1)]
            sess.run(model.train, feed_dict={model.x: batch_x, model.y_true: batch_y, model.hold_prob: 0.5})
            if i % 100 == 0:
                print('Currently on step {}'.format(i))
                print('Accuracy is:')
                # Test the Train Model
                matches = tf.equal(tf.argmax(input=model.y_pred, axis=1), tf.argmax(input=model.y_true, axis=1))
                acc = tf.reduce_mean(input_tensor=tf.cast(matches, tf.float32))
                print(sess.run(acc, feed_dict={model.x: test_images, model.y_true: test_labels, model.hold_prob: 1.0}))
                print('\n')
            saver.save(sess, save_location)


def cli_main(flags):
    model = cnn.CNN(flags.learning_rate)

    train(model, flags.epochs, flags.save_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int,
                        default=1200,
                        help='The number of epochs for training')

    parser.add_argument('--learning_rate', type=int,
                        default=0.001,
                        help='The learning rate to use during training')
    parser.add_argument('--save_location', type=str,
                        default="./model/model",
                        help='The location to save the model in')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
