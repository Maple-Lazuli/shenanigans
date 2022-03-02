import tensorflow.compat.v1 as tf
import argparse
import numpy as np

tf.enable_eager_execution()

# tf.disable_v2_behavior()

import cnn


def parse_records(file):
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    raw_dataset = tf.data.TFRecordDataset(file)
    images = []
    labels = []
    for raw_record in raw_dataset:
        img_raw = tf.io.parse_single_example(raw_record, data)['image_raw']
        img_raw = tf.io.decode_raw(tf.constant(img_raw), tf.uint8)
        images.append(img_raw)

        label = tf.io.parse_single_example(raw_record, data)['label']
        one_hot_encoded = np.zeros(10, dtype=np.uint8)
        one_hot_encoded[np.frombuffer(label, dtype=np.uint8)] = 1
        labels.append(one_hot_encoded)
    return np.array(images), np.array(labels)


def train(model, epochs, save_location, batch_size, train_data, test_data):
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.train.Saver()
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            for batch in range(len(train_images) // batch_size - 1):
                batch_x = train_images[batch_size * batch: batch_size * (batch + 1)]
                batch_y = train_labels[batch_size * batch: batch_size * (batch + 1)]
                sess.run(model.train, feed_dict={model.x: batch_x, model.y_true: batch_y, model.hold_prob: 0.5})
            remainder = len(train_images) % batch_size
            batch_x = train_images[-remainder:]
            batch_y = train_labels[-remainder:]
            sess.run(model.train, feed_dict={model.x: batch_x, model.y_true: batch_y, model.hold_prob: 0.5})

            if i % 100 == 0:
                print('Currently on epoch {}'.format(i))
                print('Accuracy is:')
                # Test the Train Model
                matches = tf.equal(tf.argmax(input=model.y_pred, axis=1), tf.argmax(input=model.y_true, axis=1))
                acc = tf.reduce_mean(input_tensor=tf.cast(matches, tf.float32))
                print(sess.run(acc, feed_dict={model.x: test_images, model.y_true: test_labels, model.hold_prob: 1.0}))
                print('\n')
        saver.save(sess, save_location)


def cli_main(flags):
    # just a hack to get around TF record reading
    train_records = "./mnist_tf/train/mnist_train.tfrecords"
    test_records = "./mnist_tf/test/mnist_test.tfrecords"
    train_images, train_labels = parse_records(train_records)
    test_images, test_labels = parse_records(test_records)
    print("finshed parsing tf records")
    tf.disable_eager_execution()
    model = cnn.CNN(flags.learning_rate)
    train(model, flags.epochs, flags.save_location, flags.batch_size, train_data=(train_images, train_labels),
          test_data=(test_images, test_labels))


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
