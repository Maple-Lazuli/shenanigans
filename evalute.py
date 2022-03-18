import tensorflow as tf
from dataset_generator import DatasetGenerator
from lenet_mnist_graph import parse_records

location = "./mnist_tf/test/mnist_test.tfrecords"
test_df = DatasetGenerator(location, parse_function=parse_records, shuffle=True,
                               batch_size=1)
iterator = test_df.get_iterator()
next_step = iterator.get_next()

# experiment with mnist since the model is small and stored in github
sess = tf.compat.v1.Session()
# TODO Figure out what the error means
saver = tf.compat.v1.train.import_meta_graph("./mnist_model/mnist.meta")
saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./mnist_model/"))

graph = tf.compat.v1.get_default_graph()

input_image = graph.get_tensor_by_name("mnist_model/X:0")
predicted_label = graph.get_tensor_by_name("mnist_model/y_pred:0")

try:
    sess.run(iterator.initializer)
    while True:
        features = sess.run(next_step)
        batch_x = features['input']
        batch_y = features['label']
        # test sucessful data export
        print(batch_y)
except tf.errors.OutOfRangeError:
    print("done")

"""
ValueError: Converting GraphDef to Graph has failed with an error: 'Cannot add function 
'__inference_Dataset_flat_map_read_one_file_11' because a different function with the same name already exists.' 
The binary trying to import the GraphDef was built when GraphDef version was 716. The GraphDef was produced by a binary
 built when GraphDef version was 987. The difference between these versions is larger than TensorFlow's forward 
 compatibility guarantee, and might be the root cause for failing to import the GraphDef.
"""