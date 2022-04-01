# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 30
# Performance
![image](images/fa7e73c69029ca127ecf566f6b96dd6d.png)
![image](images/2c5eb1e1b409e3e8cde4c9e33d68e411.png)
![image](images/4a81d78b89bd596fd18cd980c7a3c84a.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/train/satsim_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/valid/satsim_valid.tfrecords consists of 350, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/6776adbfd1f1bb4295ce978d54fe9906.png)
![image](./images/cfc190756599808102baad708fb52e0c.png)
![image](./images/a3c49ba7db74c785a06136f79687517e.png)
![image](./images/b657064851240537c6c73e7a74b4b1ae.png)
![image](./images/f650321cdaa0d1b393c29ee3ba0c91d8.png)
![image](./images/2bdf387a90e5d47d47fecc16ea714308.png)
![image](./images/4645c1a4190730da37cf9bd5d086ab17.png)
![image](./images/1032f771f2b8355b1dcd10d6db27d1ca.png)
![image](./images/5fcc6efff418ed558158c01eb5fe5d05.png)
![image](./images/921d55477cfcc2ef15710ba7deec2df4.png)
![image](./images/db59b335d227771434379a9fb9f37474.png)
![image](./images/bfb2af2027c498e4b0e7b15baf3f1bcc.png)
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
![image](images/9368b60dc28821e0a9ad10381d69341f.png)
