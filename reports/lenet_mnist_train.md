# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 1
# Performance
![image](images/6f8f29abe021d55106d5ec7bc89e5002.png)
![image](images/cc9d5f8081de61fb17d950259f767415.png)
![image](images/f5eab7b8e24a1be4db09e367d71e9f3e.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 56000, served in batch sizes of 50.

### Validation Set 
The validation set located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 7000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/b6e76558b36ad7ee9a7b809ed2d2658e.png)
![image](./images/a9e9822ef1ffbbf9dfb7d80507190861.png)
![image](./images/1ac556eda35823a0b91e75c39b06592e.png)
![image](./images/e51d4046ff40d9ffb3a230bc8f7c5ab1.png)
![image](./images/5b6b1e51b4a40bb962a846966f6c097a.png)
![image](./images/8278fd0f973cb7ef7835cc5c7a4ea5d4.png)
![image](./images/163e1a6e831bf68b3e7fd210eba7d8ee.png)
![image](./images/7213ced1382d9f1b5ed0b9deeeaa8512.png)
![image](./images/0794a58d16679bc5424fb5c374d0fa22.png)
![image](./images/f72341aa94b765591b5fa2b9aa2354a2.png)
![image](./images/76ef560ee4b8361112d00847a443b7f3.png)
![image](./images/f21f8d05d695d99a78207f612832205d.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/15782daf0c69089d5a0c84296fe3e259.png)
### Example 2 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/562390ae356bf60f08c3b94c5cb52105.png)
### Example 3 
1. height:28
2. width:28
3. depth:0
4. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/1f1ef2780ceba9084f7c6146455bba69.png)
### Example 4 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/0452d2d9c16d586a4cf3d54b195a0050.png)
### Example 5 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/a8b3ccebeff3ab49407f954428b4cb08.png)
### Example 6 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/75b869d3d50f3533ab9044657e3913d8.png)
### Example 7 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/08743f07ac24a8d9fd9c028cd5a246e6.png)
### Example 8 
1. height:28
2. width:28
3. depth:0
4. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/d9c499416d91ddf1c5eae7c30cc29cf8.png)
### Example 9 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/226c0faf291d2a1f4fb86945cdaa56e7.png)
### Example 10 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/7949d5d86d6a924b61f1af7a3aa4a2db.png)
