python make_tfrecords.py --config_json mnist_config_1_example.json --train_size 1
python lenet_mnist_graph.py --config_json mnist_config_1_example.json
python evaluate.py --config_json mnist_config_1_example.json
python make_tfrecords.py --config_json satsim_config_1_example.json --train_size 1
python lenet_satsim_graph.py --config_json satsim_config_1_example.json
python evaluate.py --config_json satsim_config_1_example.json