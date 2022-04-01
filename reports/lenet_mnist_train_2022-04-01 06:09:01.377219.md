# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 30
# Performance
![image](images/c128c55beb9ad8a96540dd716b4efa43.png)
![image](images/9b11f4f086a3347ac4a8f72dccb8c0de.png)
![image](images/9bf0c61958fa5b3486659f46f1178f6e.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 49000, served in batch sizes of 50.

### Validation Set 
The validation set located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/641056555046ddac3baac3828945f630.png)
![image](./images/dde0d9d683c23611c4cc3105b7dc836f.png)
![image](./images/4bac2fa17a4e79b98bfb255c0213851f.png)
![image](./images/8417a558cfce87ff5ddb8a9dc1c4cf6f.png)
![image](./images/e257d3be5ccc4ab1dabb29c542dc2c7d.png)
![image](./images/79cea00e275d1d818a23afd2ede03ecf.png)
![image](./images/3519bd395d7af20e442eb9c4af9acbaf.png)
![image](./images/1638534fcee0b03ec915f979b61a8616.png)
![image](./images/9ec6b3e2fe09284fbe4338b25a7137bc.png)
![image](./images/fdc8d8a7ea5c59ad73db62121c82366a.png)
![image](./images/0a955505cc0a90544f9b5a072ebc2757.png)
![image](./images/1059e7df1f264afd077c73fcab96365e.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/de06a5ed957ab6cece2bf6123f9fcb1a.png)
### Example 2 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/872b48da6c05c7d64978923f12944674.png)
### Example 3 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/2ca4bd2903bde3ef2a4cc5fdbd8b4e6b.png)
### Example 4 
1. height:28
2. width:28
3. depth:0
4. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/043a41b8aa1d840b5104c6d3766e4445.png)
### Example 5 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/5619e69f2c67589f3454ee9bc2412040.png)
### Example 6 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/be3d758d3e4a9f2befb185407cbb5e11.png)
### Example 7 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/9b95f95b9391b17b151ec281bc623d69.png)
### Example 8 
1. height:28
2. width:28
3. depth:0
4. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/d4d6a3b1516de24f36f9e5d1992f4567.png)
### Example 9 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/a7d84450e6eee271a16376c04ae1f98e.png)
### Example 10 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/dd9cc425e953aa38581e1be9cea63969.png)
