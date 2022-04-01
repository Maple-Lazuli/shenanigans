# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 20
# Performance
![image](images/bfb9e4174a0359bb6e225bc99e5bced8.png)
![image](images/3ee48d60c53f0fed1ccca9095f3f606c.png)
![image](images/f0fe0b31c32692ecaa8458cf6a203358.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/train/satsim_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/valid/satsim_valid.tfrecords consists of 350, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/d1fc9d9a289317f13c4a85e7126239c7.png)
![image](./images/1dbe3d5efdfd16b20ebf165050b90539.png)
![image](./images/ae024fd35eab4f2382ddbf8ac5a8f39d.png)
![image](./images/6116f9371d5d0748ebf34730e234e8fd.png)
![image](./images/d2174b7f075ad40635e675368d89638f.png)
![image](./images/466af39254f7f490c09d906303166b8a.png)
![image](./images/61579511630725c141d1ef5f92b12a6b.png)
![image](./images/e9b593d41e0d6f7b087891d279c1dd67.png)
![image](./images/ed9ee1bf11d8484b52e215e45b307d46.png)
![image](./images/14a1bb38ba66df412baf0c4dfc9b0033.png)
![image](./images/e24660f7e7156f809453b78e0fa7fe59.png)
![image](./images/6b185e18fe32921c87eadf1f459f59b4.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_low'
8. label:[0. 0. 1. 0. 0.]
![image](images/de4f42ce2055336e10b37909824af940.png)
