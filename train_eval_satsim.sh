python make_tfrecords.py --config_json ./exe_configs/satsim_config_1_example.json --train_size 1
python lenet_satsim_graph.py --config_json ./exe_configs/satsim_config_1_example.json --epochs 10
python evaluate.py --config_json s./exe_configs/atsim_config_1_example.json
python lenet_satsim_graph.py --config_json ./exe_configs/satsim_config.json --epochs 10
python evaluate.py --config_json ./exe_configs/satsim_config.json