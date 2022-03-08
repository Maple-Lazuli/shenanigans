import argparse
import tensorflow as tf
import make_tfrecords_satsim as sim
import make_tfrecords_mnist as mnist
import os
import shutil


def make_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def create_tfrecords(data_dir,
                     output_dir,
                     tfrecords_name,
                     datapath_to_examples_fn,
                     tf_example_builder_fn,
                     partition_examples_fn,
                     splits_dict={"data": 1.0}):
    examples = datapath_to_examples_fn(data_dir)

    partitioned_examples = partition_examples_fn(examples, splits_dict)

    for (split_name, split_examples) in partitioned_examples.items():

        print(f"Writing partition {split_name} w/ {len(split_examples)} examples")

        # build a clean directory to store the partition in
        partition_output_dir = os.path.join(output_dir, split_name)
        make_clean_dir(partition_output_dir)

        tfrecord_name = tfrecords_name + "_" + split_name + '.tfrecords'
        output_path = os.path.join(partition_output_dir,
                                   tfrecord_name)

        with tf.io.TFRecordWriter(output_path) as writer:
            for example in split_examples:
                tf_example = tf_example_builder_fn(example)
                writer.write(tf_example.SerializeToString())


def main_cli(flags):
    split_dict = {"train": 0.8, "valid": 0.1, "test": 0.1}

    if flags.parser == "satsim":
        partition_fn = sim.partition_examples
        build_tf_dataset = sim.build_tf_dataset
        build_tf_example = sim.build_tf_example

    elif flags.parser == "mnist":
        partition_fn = mnist.partition_examples
        build_tf_dataset = mnist.build_tf_dataset
        build_tf_example = mnist.build_tf_example

    else:
        raise ValueError("Unknown Record Parser")

    create_tfrecords(data_dir=flags.data_dir,
                     output_dir=flags.output_dir,
                     tfrecords_name=flags.name,
                     datapath_to_examples_fn=build_tf_dataset,
                     tf_example_builder_fn=build_tf_example,
                     partition_examples_fn=partition_fn,
                     splits_dict=split_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str,
                        default="satsim",
                        help='Name of the dataset to build.')

    parser.add_argument('--data_dir', type=str,
                        default="./generated_data",
                        help='Path to raw data.')

    parser.add_argument('--output_dir', type=str,
                        default="./generated_data_tfrecords",
                        help='Path to the output directory.')

    parser.add_argument('--parser', type=str,
                        default="satsim",
                        help='Path to the output directory.')

    flags, unparsed = parser.parse_known_args()

    main_cli(flags)
