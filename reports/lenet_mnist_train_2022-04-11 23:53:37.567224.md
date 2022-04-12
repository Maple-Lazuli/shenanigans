# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 10
# Performance
![image](images/d7490e26e89c9af684b59a7c2d5fa78c.png)
![image](images/3474bd7385bd4cfb22b94628eb9b4c71.png)
![image](images/489943908a9aefa64065c0de6bba7e58.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 49000, served in batch sizes of 50.

### Validation Set 
The validation set located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/ad133081cec25c6e7c28aa6c81e4b15f.png)
![image](./images/cf827ffa84c796898d715cace012755a.png)
![image](./images/3f3563b8cd666f4d9738c7f0c4f2ce58.png)
![image](./images/b4d080d365fe7fd7184b5dffb1d6a115.png)
![image](./images/b067a73d78d4ad9f0329d1f8c912edd5.png)
![image](./images/070f145730dfc05a90bcf167d8076c63.png)
![image](./images/e1a17f2fbab7037d962fa60fd34264f0.png)
![image](./images/c093e7900b7f1be76ef1448d57d7182f.png)
![image](./images/a79dbbaed7b178e58fbf029ae0cf6082.png)
![image](./images/df677165eae567b0e0fcb0cc818edb6a.png)
![image](./images/f3afbf9df329b52baf3bfb65d20d4943.png)
![image](./images/d9befdedbed85e5a423dfbcc86d100bc.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:28
2. width:28
3. depth:0
4. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/748378da3340037b17a797e425e8f647.png)
### Example 2 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/02cb05e28ca7e971f727fc51ddc06927.png)
### Example 3 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/5a4df9f7fc89ed8138fbc1dded1b764b.png)
### Example 4 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/dafd871e2e5dcf48a4f090ff3b738e50.png)
### Example 5 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/bacbc399e56bf60bd3ddf68d5e4f6cf4.png)
### Example 6 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/2859ef0ccb47919392b0b83d12dd4527.png)
### Example 7 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/249d3b2f6919be4bb384f6cb1d7df570.png)
### Example 8 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/3971863c6427a7cb3704423c6abefe2e.png)
### Example 9 
1. height:28
2. width:28
3. depth:0
4. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/779cff69f60aa729ec5d9a6e3731c6dc.png)
### Example 10 
1. height:28
2. width:28
3. depth:0
4. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/739e595abf3184416a77b22d8e58d6fb.png)
