# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 30
# Performance
![image](images/3b86366c08191ef84d018293c0d11e7d.png)
![image](images/388fc901e261ff444e32385924b02e57.png)
![image](images/9f46100d358ab2b93d2218fafef5c7b8.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf_1_train_example/train/mnist_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/90859177af79f9dbbf5ddbacbed78806.png)
![image](./images/ee70fc25dd25c708ce7d0c4722e0f079.png)
![image](./images/a4fb4163617f5534384be7ce0b70cc39.png)
![image](./images/005bfdefe0fdec3f36b889e2e1017430.png)
![image](./images/1435be52b5b97e9cb1585b0b17dd8d39.png)
![image](./images/5a6d4ef3062d1f417fb767fcec277b31.png)
![image](./images/f41a9ae004b3921bf6e4b04bb6f0a652.png)
![image](./images/7493c17057eb9f5b04e9f06774d88316.png)
![image](./images/bc458b719bcc06c17da03b6aace10d00.png)
![image](./images/d410b0cd1807af2ac6a2ff42b25fcccb.png)
![image](./images/26499fb585ffd8f5c111123312856f1a.png)
![image](./images/9f9326895a7e3e0064b29b2ddaccc4a6.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/f0903d5e604efc883d3f308e8461a666.png)
