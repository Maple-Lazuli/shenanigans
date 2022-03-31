import argparse
import tensorflow as tf
import make_tfrecords_satsim as sim
import make_tfrecords_mnist as mnist
import os
import shutil
import json


def get_parameters(file_name):
    with open(file_name) as json_file:
        parameters = json.load(json_file)
    return parameters


def make_clean_dir(directory):
    """
    Clears the directory for re-writing

    Parameters
    ----------
    directory: The directory to purge

    Returns
    -------
    None
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def create_tfrecords(data_dir,
                     output_dir,
                     tfrecords_name,
                     datapath_to_examples_fn,
                     tf_example_builder_fn,
                     partition_examples_fn,
                     splits_dict={"data": 1.0},
                     train_size=-1):
    examples = datapath_to_examples_fn(data_dir)

    partitioned_examples = partition_examples_fn(examples, splits_dict)

    for (split_name, split_examples) in partitioned_examples.items():
        if split_name == "train" and train_size != -1:
            print(f"Writing partition {split_name} w/ {train_size} examples")
        else:
            print(f"Writing partition {split_name} w/ {len(split_examples)} examples")

        # build a clean directory to store the partition in
        partition_output_dir = os.path.join(output_dir, split_name)
        make_clean_dir(partition_output_dir)

        tfrecord_name = tfrecords_name + "_" + split_name + '.tfrecords'
        output_path = os.path.join(partition_output_dir,
                                   tfrecord_name)

        with tf.io.TFRecordWriter(output_path) as writer:
            if split_name == "train" and train_size != -1:
                for idx, example in enumerate(split_examples):
                    if idx >= train_size:
                        break
                    else:
                        tf_example = tf_example_builder_fn(example)
                        writer.write(tf_example.SerializeToString()) if tf_example is not None else None
            else:
                for example in split_examples:
                    tf_example = tf_example_builder_fn(example)
                    # only write if the tf_example_builder_fn does not return a none value
                    writer.write(tf_example.SerializeToString()) if tf_example is not None else None


def main_cli(flags):

    config_dict = get_parameters(flags.config_json)

    if config_dict['graph_name'] == "satsim":
        partition_fn = sim.partition_examples
        build_tf_dataset = sim.build_tf_dataset
        build_tf_example = sim.build_tf_example

    elif config_dict['graph_name'] == "mnist":
        partition_fn = mnist.partition_examples
        build_tf_dataset = mnist.build_tf_dataset
        build_tf_example = mnist.build_tf_example

    else:
        raise ValueError("Unknown Record Parser")

    create_tfrecords(data_dir=config_dict['source_data_dir'] ,
                     output_dir=config_dict['tf_record_output_dir'] ,
                     tfrecords_name=config_dict['graph_name'],
                     datapath_to_examples_fn=build_tf_dataset,
                     tf_example_builder_fn=build_tf_example,
                     partition_examples_fn=partition_fn,
                     splits_dict=config_dict['split_dict'],
                     train_size=flags.train_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_json', type=str,
                        default='./mnist_config.json',
                        help="The config json file containing the parameters for the model")

    parser.add_argument('--train_size', type=int,
                        default=-1, # if size < 0, parameter is ignored
                        help="The number of train examples to use from the data source")

    flags, unparsed = parser.parse_known_args()

    main_cli(flags)
