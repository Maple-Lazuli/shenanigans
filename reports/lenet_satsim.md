# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 3
# Metrics
![image](images/f4e2bb710d75dab1205fd7a97b307725.png)
![image](images/de7391dc00e596ab03e78865eade9366.png)
![image](images/ff5a788814bafac2528815c97d8d7fc3.png)
# Datasets 
### Training Set 
The training set located at ./generated_data_tfrecords/train/satsim_train.tfrecords consists of 1500, served in batch sizes of 50.
### Testing Set 
The testing set located at ./generated_data_tfrecords/test/satsim_test.tfrecords consists of 150, served in batch sizes of 50.
### Test Set / Training Set Comparison 
This section compares the contents of the test and train sets used.
![image](./images/2e8bbf3b56090709670bdbcfed8a9c62.png)
![image](./images/3d5954fa4d14b0dde00d90aa0973e85e.png)
 ![image](./images/8d27e9c2bcc331ae893b189344ef547a.png)
![image](./images/74000af466662d3fc5e217760d8c260b.png)
 ![image](./images/0c4314874425fed7be0d7118106ee46f.png)
![image](./images/06dd2baccbc97644a1bae215911322ae.png)
 ![image](./images/8669f4cf72fa20ddced71400a7c89df4.png)
![image](./images/4905391db19b1f700a32522be2b317cf.png)
 ![image](./images/f230d5c1278e06e4e171a0adca18eb69.png)
![image](./images/b1ae010df13eae6c21d6110ba6b14ae2.png)
 ![image](./images/7ea5a7255a5273c9298ebd026cf34661.png)
![image](./images/984aefa667b371cd633f3668bfb65117.png)
 ![image](./images/64f06b45aadb4447fc0ac3e881f5220f.png)
![image](./images/c31c2ce3cc4b09e9fa56961ad098a11a.png)
 ![image](./images/7a5cd916824e67d213480d33f842a3e1.png)
![image](./images/b5f81e72ee429bf0f3838c87634536ba.png)
 ![image](./images/119b3eeb60c6450f2e90829420213441.png)
![image](./images/2e352fa8c7dd0fc8e5c938defe917834.png)
 ![image](./images/7b5374418f567bd3821acbb54daeac21.png)
![image](./images/982bf905ad321ea102f28a6221d8f219.png)
 ![image](./images/40246e885cd452e7599b48b205b6f5d5.png)
![image](./images/8b3146974ac5f8ae48ef3064864910ef.png)
 ![image](./images/b62cf7f10c5bb8ae89f2a6fa58940574.png)
![image](./images/36a25f14819359e01f41d6a3a83d3262.png)
 ![image](./images/1c71c9b0f91b8229e4960ca558ff37f4.png)
![image](./images/d175b7d88f64c5016e15446cebfca0a5.png)
 ![image](./images/d23da9611f55ceaaf4a58328dc0b6ad4.png)
![image](./images/1ab2372dfe29628eab8a19fc906a29d6.png)
 # Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 10. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'collision_high'
7. label:[0. 1. 0. 0. 0.]
![image](images/c34077afae2e08a73e8081693ec0cefc.png)
### Example 20. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'collision_low'
7. label:[0. 0. 1. 0. 0.]
![image](images/6f41268875ce379326e13280c4a01367.png)
### Example 30. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'breakup'
7. label:[0. 0. 0. 0. 1.]
![image](images/867d39ef89de0b8be42ce9ca24dfdb65.png)
### Example 40. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'rpo'
7. label:[0. 0. 0. 1. 0.]
![image](images/f3d9c15346f450ad7b495250c9059234.png)
### Example 50. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'nominal'
7. label:[1. 0. 0. 0. 0.]
![image](images/8a711a424db65c71949f8fc6414a38c9.png)
