# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 1
# Peformance
![image](images/983e87ac0e74d39c4a0c929439937fe4.png)
![image](images/d235c7c2c753949cf6b52c81127468cb.png)
![image](images/a6d55ed8c89b96db02cb6aba0bd2ff2b.png)
# Datasets 
### Training Set 
The training set located at ./generated_data_tfrecords/train/satsim_train.tfrecords consists of 1500, served in batch sizes of 50.
### Testing Set 
The testing set located at ./generated_data_tfrecords/test/satsim_test.tfrecords consists of 150, served in batch sizes of 50.
### Test Set / Training Set Comparison 
This section compares the contents of the test and train sets used.
![image](./images/0e2bd154dbab4555c0fd54f687650d76.png)
![image](./images/7d2b9bed5701c6a5c1271bea39981b05.png)
 ![image](./images/41c03c225ddfc5ac9170ba34bcf99127.png)
![image](./images/05685666bd4f481f51d719594f58ca55.png)
 ![image](./images/8adba9d4c8e71457efd009fe34ea72dd.png)
![image](./images/33ace36eafd2d3b05a85cb833d4ca505.png)
 ![image](./images/f02bbec3985efb08f6716082ae83513a.png)
![image](./images/c3707c4c2a982a6242dffe9cc50c1e0d.png)
 ![image](./images/90a8557de75e40358a8d7bf89744bafc.png)
![image](./images/78d27f3e302539c348bb77028c2b9574.png)
 ![image](./images/b7674a95593915041e57749d5442698c.png)
![image](./images/8f5014db35e062f33b5c8ec510bf8374.png)
 ![image](./images/12e205d8adcd1957c8c869ae621f6e4b.png)
![image](./images/c10d430d8f986e0f18501d9d3ca1a7ce.png)
 ![image](./images/a6836ce2c6aa64f091dd277fe6a1494d.png)
![image](./images/c3fd4e85a6d11906a881c3422352f9d9.png)
 ![image](./images/d790d93f32b1eee99a893f4ce115c99f.png)
![image](./images/a73a512a70377b4bd2857e7dbccb8b0a.png)
 ![image](./images/f2a0122210e61898a2f5a2e15ab2803d.png)
![image](./images/5f30f2618a16cee3824ead4e1bef7cc4.png)
 ![image](./images/844723c38ababc6482354919c3690673.png)
![image](./images/2aa4f4952f069901831117c81c28b3ec.png)
 ![image](./images/9897a8adf28af95f215576a36b85e977.png)
![image](./images/6173956df7ac99f6f2f3344c39fcf8e6.png)
 ![image](./images/6d811c47a53a3f9983b4eb1ef37071f1.png)
![image](./images/4ba4e3b52b8297f9338aa4edd01f3cf6.png)
 ![image](./images/82fec14cdd604916f39cfe2141a51764.png)
![image](./images/930a40663a77262c913c990ce44db345.png)
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
![image](images/80bbb9a48934a49733543ccd88040ac7.png)
### Example 20. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'breakup'
7. label:[0. 0. 0. 0. 1.]
![image](images/d1ab310a8f5d8119567675bf232d194e.png)
### Example 30. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'nominal'
7. label:[1. 0. 0. 0. 0.]
![image](images/f74f915b707cab99959519cc0e1bcd31.png)
### Example 40. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'collision_low'
7. label:[0. 0. 1. 0. 0.]
![image](images/04c69b304a0874ff6e8b9bdc2561f326.png)
### Example 50. height:512
1. width:512
2. depth:16
3. field_of_view_x:0.0006727635045535862
4. field_of_view_y:0.0006727635045535862
5. stray_light:0
6. class_name:b'rpo'
7. label:[0. 0. 0. 1. 0.]
![image](images/59a0666fc96fb5e68e55e8d25f8950e1.png)
