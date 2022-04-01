# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 20
# Performance
![image](images/de0e9de956094ef47724a46ac91067bd.png)
![image](images/82c7480a573635991a31793d724b326b.png)
![image](images/218a6d331870b06544c09bcf14967a03.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 49000, served in batch sizes of 50.

### Validation Set 
The validation set located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/db98219878819f3891c4ad9f70b9e618.png)
![image](./images/f8c6621a086c6144966db6669f23c563.png)
![image](./images/e0e0d9fff5d2a47051b29859c23f9cbb.png)
![image](./images/c9d78590a4d1a27a7c143d7a8aceaaa9.png)
![image](./images/6dc570c6befa033939253bd6f318b128.png)
![image](./images/f8792fad1e7c371f1415c192ec396378.png)
![image](./images/5e920c275cac744c7fed370a13bb9eaf.png)
![image](./images/fb76bfb209c3089368f57f16c849b226.png)
![image](./images/d96a38caba3e7c406f86112dbe8d7264.png)
![image](./images/a0bd0b194a3f28e1c0f8235c664622ad.png)
![image](./images/ce28febd7feaca7553997ef9106e87db.png)
![image](./images/a88f0f92bf8536ec81dc43430708d918.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/b7f156fa7f339c5edbfe6d088a84a7ee.png)
### Example 2 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/65533a873e1c3b0c3b357aaeccd0d6f3.png)
### Example 3 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/ccd80f11b2e124794b4e3ad44c69a5fd.png)
### Example 4 
1. height:28
2. width:28
3. depth:0
4. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/2f5a9951842496ff0b303d9953f47b2b.png)
### Example 5 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/629d5a2c1beaa033085aee0e91838177.png)
### Example 6 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/6bcf083dd28ab2a5486de601616978af.png)
### Example 7 
1. height:28
2. width:28
3. depth:0
4. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/5dd18d71672b017522cae4f7c953e785.png)
### Example 8 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/ec6af99c623e6d19141bc586b67e57fa.png)
### Example 9 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/3f66e3b904172b22cd79ee3061858134.png)
### Example 10 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/90092fe8d398513a8c53a5f4777d4241.png)
