# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 4
# Performance
![image](images/26710f881ba56660da3e23caed7b2d1a.png)
![image](images/8af6be0fe45a1dd9146def07ea1eed2c.png)
![image](images/2ea349f33ac9867595e8515ed50ef9a2.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 49000, served in batch sizes of 50.

### Validation Set 
The validation set located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/6e2d232672fe27e10b7c0f050fa2bfe0.png)
![image](./images/f32cf148ab7b82db743dc6ffbd3ed930.png)
![image](./images/221f03d6145401f287347a2e5a382b50.png)
![image](./images/942168861b68edd89cb64938676dcdd6.png)
![image](./images/e7861d457961877d2cc95f99a15d4c69.png)
![image](./images/10462de428d80ae4e380c7d51a8a330e.png)
![image](./images/8154001735f71cb343949d6a72812dd2.png)
![image](./images/529b29ac1f7998c0ede40f712fc9ecb4.png)
![image](./images/f3bd064f0522a7b62dd935d427fc236a.png)
![image](./images/00cddbff78fc093ee5a13e460076effa.png)
![image](./images/5b37135dc28760ec6225c1e7201e10d1.png)
![image](./images/3ffffc93b24cbfbdae8fc63a7af32ee3.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/1b7acce8dc48e5e9e13192a78f014be2.png)
### Example 2 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/028673fe43870cea754dc907c8c7330d.png)
### Example 3 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/94a902230cfbddbc4e82107ea6058046.png)
### Example 4 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/b7bffa84cf2ce8b55391a14d2dfc6050.png)
### Example 5 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/ac9f192902d151fe638fd41d29f19622.png)
### Example 6 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/61d22ea1aca55197b50f78209c9ffccc.png)
### Example 7 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/5513b30262bfb3f9f782b2d1ae92b10c.png)
### Example 8 
1. height:28
2. width:28
3. depth:0
4. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/ee7a1ddbd87a4038dc86fe07a30bccfa.png)
### Example 9 
1. height:28
2. width:28
3. depth:0
4. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/2b5cbd7122532ae0cbd20cee767838ab.png)
### Example 10 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/1f05f9d8caf474dbb0f528860714524d.png)
