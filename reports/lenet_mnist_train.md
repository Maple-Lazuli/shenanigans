# Overview 

        Implementation of a variant of lenet to classify handwritten digits between 0 and 9. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 20
# Peformance
![image](images/3f29fe2c4d11a37758f68847be801111.png)
![image](images/d937143d57bc2076f9bff61fc3fdda3e.png)
![image](images/03ec1282326eb01235db7502c58fe083.png)
# Datasets 
### Training Set 
The training set located at ./mnist_tf/train/mnist_train.tfrecords consists of 56000, served in batch sizes of 50.
### Testing Set 
The testing set located at ./mnist_tf/test/mnist_test.tfrecords consists of 7000, served in batch sizes of 50.
### Test Set / Training Set Comparison 
This section compares the contents of the test and train sets used.
![image](./images/9ab736c7a2ab1e6081288fbd0aea77ee.png)
![image](./images/0800281366396a79e2212e9e00e1d866.png)
 ![image](./images/8ad18ba9b6fe0708966666528bb23cfa.png)
![image](./images/c37e9f52b0ca3517680010eee5756aa2.png)
 ![image](./images/f3130687eed928921377b65a957ab17d.png)
![image](./images/cc6059aa079c61abf3624b48e06972d0.png)
 ![image](./images/7d50b2fb44f8e3aea9460779f86e02d2.png)
![image](./images/eff264e611d6d8517dc2c8fb464c0122.png)
 ![image](./images/02cf83340cdbb7241f37cc4afe67520f.png)
![image](./images/210f2389db22a0497e6912b04911dff1.png)
 ![image](./images/fa0851cd2a7f4cb67759460d410b975c.png)
![image](./images/473732f9107fce97dda75d19468f3538.png)
 # Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
0. height:28
1. width:28
2. depth:0
3. label:[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/d497d71a2203c819b1cc2c4e8bad6732.png)
### Example 2 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
![image](images/902fbfd30c7c4b2c659f76df6e6abfb4.png)
### Example 3 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
![image](images/b8626b72b99b629460d64213d1e1e6c5.png)
### Example 4 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
![image](images/49ff9e9b775fb8e536bfc565a8248b25.png)
### Example 5 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
![image](images/d8dd0629169d1e452209456381f9ab7e.png)
### Example 6 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
![image](images/187f4943f83b3ee31a872361d7574c17.png)
### Example 7 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
![image](images/3a9da654317e5c2dba7fb2d21d2bda26.png)
### Example 8 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
![image](images/d9b58640b20240fe7ad8751730aa11f8.png)
### Example 9 
0. height:28
1. width:28
2. depth:0
3. label:[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
![image](images/d7ba0feca66937de4311746d6f2ea7a9.png)
### Example 10 
0. height:28
1. width:28
2. depth:0
3. label:[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
![image](images/7244d6b3cca5910b28187451113a4d46.png)
