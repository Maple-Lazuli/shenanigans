import tensorflow.compat.v1 as tf
import argparse
import numpy as np

tf.enable_eager_execution()


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


def evaluate(save_location, val_data=[None, None]):
    init = tf.compat.v1.global_variables_initializer()
    val_images, val_labels = val_data
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph(save_location + ".meta").restore(sess, tf.train.latest_checkpoint('./model'))
        print('Accuracy is:')
        y_pred = tf.tf.get_default_graph().get_tensor_by_name("y_pred")
        y_true = tf.get_tensor_by_name("y_true")
        x = tf.get_tensor_by_name("x")
        hold_prob = tf.get_tensor_by_name("hold_prob")

        matches = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))
        acc = tf.reduce_mean(input_tensor=tf.cast(matches, tf.float32))
        print(sess.run(acc, feed_dict={x: val_images, y_true: val_labels, hold_prob: 1.0}))
        print('\n')

def cli_main(flags):
    # just a hack to get around TF record reading
    val_records = "./mnist_tf/train/mnist_train.tfrecords"
    val_images, val_labels = parse_records(val_records)
    print("finished parsing tf records")
    tf.disable_eager_execution()
    evaluate(flags.save_location, val_data=[val_images, val_records])
    evaluate(flags.save_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_location', type=str,
                        default="./model/model.ckpt",
                        help='The location to save the model in')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(parsed_flags)
