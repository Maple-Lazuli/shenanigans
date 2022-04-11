python make_tfrecords.py --config_json ./exe_configs/mnist_config_1_example.json --train_size 1
python lenet_mnist_graph.py --config_json ./exe_configs/mnist_config_1_example.json --epochs 10
python evaluate.py --config_json ./exe_configs/mnist_config_1_example.json
python lenet_mnist_graph.py --config_json ./exe_configs/mnist_config.json --epochs 10
python evaluate.py --config_json ./exe_configs/mnist_config.json