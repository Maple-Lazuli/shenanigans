# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 30
# Performance
![image](images/5cf9d9a7956fb605b70958942ec12869.png)
![image](images/aa2ed8043a57d1bee7babf4bdcef40a2.png)
![image](images/f282ffff841623df1b937114b054c3fe.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 56000, served in batch sizes of 50.

### Validation Set 
The validation set located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 7000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/93868e08288fdc7d6d333ebff9653972.png)
![image](./images/e80a47b5c704b5a3f0bb082da8a896f4.png)
![image](./images/455afd09ed29bce00a1b4a1ab028626b.png)
![image](./images/b3eb87c34122509216e5fbbf64676f70.png)
![image](./images/bdd5f2745b826ebd450d42f1dcc43c83.png)
![image](./images/9c34c860423931045011ff1749fe3109.png)
![image](./images/0d0fb91fbbd24f51b51e63e09270b3cb.png)
![image](./images/baefa8d9357bde74eac6c9c83210f8de.png)
![image](./images/a76622b66dc804916035fbd0a63df938.png)
![image](./images/4d776f147ba31ed5e0ee80d2782705f0.png)
![image](./images/8232bb751ecc9fc454421350adf6cfb4.png)
![image](./images/fa0d7a465641526845c8d42af52a4176.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/75ae9e2c63104129a970785d7c3d5b71.png)
### Example 2 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/373762a8f47126f18e9ea7e35b785ffc.png)
### Example 3 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/e9576fd66905906a6f7c60a7e91741b0.png)
### Example 4 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/5e33e18cf56fec82ea2e373e7f6c1dd9.png)
### Example 5 
1. height:28
2. width:28
3. depth:0
4. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/0c34d6292baca747e73e742ba52445b6.png)
### Example 6 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/b3a900d57bf802568d924d5027126ab0.png)
### Example 7 
1. height:28
2. width:28
3. depth:0
4. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/b1ef5669c22615f293bd77dd6359f009.png)
### Example 8 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/51904ee3677aea5d30bc0527c5d5c5cd.png)
### Example 9 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/c325956df128726d73395e6bf12de32f.png)
### Example 10 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/6e54e89208a247f44c64fc16106b9264.png)
