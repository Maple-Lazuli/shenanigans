# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 10
# Performance
![image](images/5d9b77425d7bbfad8ca960e316e418d3.png)
![image](images/423a7414772f8501643b5aca26272753.png)
![image](images/981d72eb448a0780f9bcb0bc1c57d61d.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/train/satsim_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/valid/satsim_valid.tfrecords consists of 350, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/57a395094e862b13edb08593488d1d0a.png)
![image](./images/1585bf68a1a3c424fae5e6d28c2b3ed1.png)
![image](./images/7ceaff9427675fd701d5f4b3da35e600.png)
![image](./images/2547d90e09a0dc709f28acd809a0cb76.png)
![image](./images/82e132d3fba91828f226425165ca2bf6.png)
![image](./images/b5681d9298ef68a23a521f87ff38217d.png)
![image](./images/5b7509a71fedf3025d47c549cb644653.png)
![image](./images/6b896b1459a941d108a6b273358ee29e.png)
![image](./images/0dc9feec45181ed21c5025e5de2f4471.png)
![image](./images/cd4daa4f1f7231637bbb7e8950b8ec8d.png)
![image](./images/456d75c228d3c4b0dc87624b46801597.png)
![image](./images/76db12980371bc8fcb5e7374d89cfe27.png)
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
![image](images/f7848c0621c824cacb5ef8acdf2127fb.png)
