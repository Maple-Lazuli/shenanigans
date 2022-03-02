import tensorflow as tf
import numpy as np


class DatasetGenerator(object):
    def __init__(self,
                 tfrecord,
                 parse_function,
                 shuffle=False,
                 batch_size=1,
                 num_threads=1,
                 buffer=30):
        self.num_examples = None
        self.input_shape = None
        self.batch_size = batch_size
        self._parse_function = parse_function
        tfrecords_list = [tfrecord]

        self.dataset = self.build_pipeline(
            tfrecords_list,
            shuffle=shuffle,
            batch_size=batch_size,
            num_threads=num_threads,
            buffer=buffer)

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.

        :return: the number of batches that will be produced by this generator.
        """
        return int(np.ceil(self.num_examples / self.batch_size))

    def get_dataset(self):
        return self.dataset

    def get_iterator(self):

        return tf.compat.v1.data.make_initializable_iterator(self.dataset)

    def build_pipeline(self,
                       tfrecord_path,
                       shuffle,
                       batch_size,
                       num_threads,
                       buffer):
        data = tf.data.TFRecordDataset(tfrecord_path)

        data = data.map(self._parse_function, num_parallel_calls=num_threads)

        # Augment here if needed

        # Mutate here if needed

        if shuffle:
            data = data.shuffle(buffer_size=buffer)

        if batch_size > 0:
            data = data.batch(batch_size, drop_remainder=True)

        data.prefetch(buffer_size=buffer)

        return data
