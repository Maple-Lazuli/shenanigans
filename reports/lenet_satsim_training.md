# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 20
# Peformance
![image](images/5e0d9fbd4d9111b27db172df300175e2.png)
![image](images/f813c4d77bc830d2202410dd9d85b3a5.png)
![image](images/7bf43de7282197f5cf516365d56553b8.png)
# Datasets 
### Training Set 
The training set located at ./generated_data_tfrecords/train/satsim_train.tfrecords consists of 1500, served in batch sizes of 50.
### Testing Set 
The testing set located at ./generated_data_tfrecords/test/satsim_test.tfrecords consists of 150, served in batch sizes of 50.
### Test Set / Training Set Comparison 
This section compares the contents of the test and train sets used.
![image](./images/63fd34d47206270166ce74d57117f6c2.png)
![image](./images/626a48dfde963fa8be9ec74d2182c1b0.png)
 ![image](./images/a040bd5cfd444ee3b8626baa271b945d.png)
![image](./images/84e57977a667a18cbf79e520c1ea4705.png)
 ![image](./images/59e6851acb760d40d01a0dfe9e6c677b.png)
![image](./images/670427dbb03633304229c3214ef1fa06.png)
 ![image](./images/d69880746adc9391ba837ef65b44b79f.png)
![image](./images/ab269de8026ac078bee7975f39c6ef01.png)
 ![image](./images/6b3b83d270dc689fb86272729dea28d2.png)
![image](./images/a2d25973f1611c992755482b375a8add.png)
 ![image](./images/9982fb1aed449796778d095c55509001.png)
![image](./images/cffc815529692ede26ee4c4bb3ad1dac.png)
 ![image](./images/a3fd369f007e1c77f631129289acd8b2.png)
![image](./images/6eabd8f4f730abd0eeff745056039da7.png)
 ![image](./images/ef00f26bbf5dfc2d23fb3aa34f18a7a4.png)
![image](./images/6f62917c582ba848193a973fd73e683e.png)
 ![image](./images/537b3c39d47b10e39069eff453a4cdfb.png)
![image](./images/97a3fcda9e6a51776059bbacbde5c572.png)
 ![image](./images/379496728cb86b90a24a1a47e20d2262.png)
![image](./images/5b13708339c7657b4afeaedef011beb8.png)
 ![image](./images/2924d7b641d86d39368deaf5a9ba19b4.png)
![image](./images/9175dc77d0f6a85a96d3a40a37ab2e5f.png)
 ![image](./images/433cede6e4cfc185bbb8c424b623de49.png)
![image](./images/bb503dd494a02dcf877b6de278340cd9.png)
 ![image](./images/5dcf4757a383a6ab130a6382026b787b.png)
![image](./images/9a87094685fe599b6ee694af9e02fde7.png)
 ![image](./images/4c93444041536413cf6c25c2c889fb3d.png)
![image](./images/af6d4f27f5dbabcf8e0fab19a4ee552c.png)
 # Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 10. 
1. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'breakup'
7. label:[0. 0. 0. 0. 1.]
![image](images/b68e6f191766c3a390b884f2921c96c0.png)
### Example 20. 
1. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'collision_high'
7. label:[0. 1. 0. 0. 0.]
![image](images/2fdae634cfc6512e5cfa6b1cad6890dd.png)
### Example 30. 
1. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'nominal'
7. label:[1. 0. 0. 0. 0.]
![image](images/b9c0aa5ddd04198671101ec63412bbf1.png)
### Example 40. 
1. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'collision_low'
7. label:[0. 0. 1. 0. 0.]
![image](images/6805042b89b9c4fb1549b0514f4c84e4.png)
### Example 50. 
1. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'rpo'
7. label:[0. 0. 0. 1. 0.]
![image](images/6d56770ee4f70ff147851ad3095c29d0.png)
