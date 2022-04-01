# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 20
# Performance
![image](images/2d9ea6f7c48bc78f682d68cc942e94b1.png)
![image](images/425c013bbf6dc792f594cbf45a8a6447.png)
![image](images/200ca86344c4891e457a1810cda5c088.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf_1_train_example/train/mnist_train.tfrecords consists of 1, served in batch sizes of 1.

### Validation Set 
The validation set located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/ab2846f8c6885d4bfd9a9acb17958c56.png)
![image](./images/62215d561c3a769953a51da69bfe91d2.png)
![image](./images/c77be60bb568ece37395b9a843c513ec.png)
![image](./images/86513a8a494cd5f26d31b434a0759b27.png)
![image](./images/9d6ba575d4068f49cdb910263230cc30.png)
![image](./images/8509d0cc37810b667219ed44a0ddf92f.png)
![image](./images/321cb07506b4328efde3cf91a635e2f1.png)
![image](./images/8bd0b0bc9b258353b2508d00f3a1f25e.png)
![image](./images/7a146fd7569a7494bad7ae01cf3aca69.png)
![image](./images/8071d759965a1c06ff48e86ab5fd6c07.png)
![image](./images/e052c7fd5ef5972d72b0700129112697.png)
![image](./images/3ebec0842efcbe46359b94be8ee91443.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/ae5405dbe89a25d78a9477e88609d183.png)
