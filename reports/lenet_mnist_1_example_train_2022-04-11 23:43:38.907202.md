# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 10
# Performance
![image](images/2f9154ee548fcd9aa65f42e703d35cbc.png)
![image](images/e8e6df958dc59ba6d1bf3e2878033348.png)
![image](images/b07fc1c53acd4844b0613407aae667a5.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf_1_train_example/train/mnist_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/61f56dc86dcbd76823a58fc368cb612b.png)
![image](./images/c3635327e9a1c91b0614df17af9e8b67.png)
![image](./images/673d6c1325f94e323ec77108337c8a5c.png)
![image](./images/e989ad030ab070d403c84bb83199e2c6.png)
![image](./images/b08e3b26479447e08d3b60a21d484559.png)
![image](./images/c3b95e3213bb25eeb4cf1ca10f0ef186.png)
![image](./images/9260cc211268e15ddcb29d328ff2c320.png)
![image](./images/e11b88e0a1cf0c64df557e3a652b3c19.png)
![image](./images/9ad6ae63ddbbe66358cd509d2b15f8b2.png)
![image](./images/37787e1d050666fb33f5d03bb96a6797.png)
![image](./images/56b8605180b2ea64f007877991789027.png)
![image](./images/b459b79236840eedf1bbae30e6dad3c2.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/4ac853850a0533f36ca3c7bed9eb8420.png)
