# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 4
# Performance
![image](images/9e79e99c88c3aeed29601bb6535469d9.png)
![image](images/03e27ff8b8f01f1f3dc92a1463f5e70e.png)
![image](images/bd86ff13e0dd1b7af2d7026d563f9e7b.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf_1_train_example/train/mnist_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/fbb92b0e43682edea7584928bd02f60e.png)
![image](./images/cd3ad053da3f1d9f018404fe6782ba11.png)
![image](./images/079e9195d4d65138b7ff964169d7238e.png)
![image](./images/4b4c077e8cb8865694992a61de670035.png)
![image](./images/96635e5c7c836c3e0defd64259a0a08a.png)
![image](./images/a0742f8efc149ae588871b9239bf34a3.png)
![image](./images/a1c74483596d06f2ce32afd06af40a98.png)
![image](./images/e3370e6a58a8db05e0ddcd8289fb0f1c.png)
![image](./images/4a6d9a1dc9d5c845ce5ac2a0497f48fc.png)
![image](./images/0ecc47586d9b888770c7d1a7bdf69c7f.png)
![image](./images/ceb0d2f9d6a3579eed527fdb8e3264d6.png)
![image](./images/439c13393c64da2eaf803ccea53a4e3f.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/71ed1f4af00cd19459e79384ed80da84.png)
