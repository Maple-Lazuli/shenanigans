# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 10
# Performance
![image](images/801b1c52e69665f6bc9cb5c049c46959.png)
![image](images/6b0a0fc854c0c79ae8f52739b6b59f62.png)
![image](images/097047645093145e64e3c6f235a7bdd0.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/train/satsim_train.tfrecords consists of 1200, served in batch sizes of 50.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 350, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/b8e8491cc973f9d305fdac8da5171af8.png)
![image](./images/74ab701fd0e423dc30330f7605643334.png)
![image](./images/76a368b24aef3417bdcc67910027225e.png)
![image](./images/562db1689968e19b6ec8930ce9a25ef3.png)
![image](./images/5f53bc459b22da09497009073f1a3e7f.png)
![image](./images/820e61860f05e47bb3635f4de703a53a.png)
![image](./images/934f633fefa905190177e4c1c3f81688.png)
![image](./images/798c3dda5f23fe11e17154ad7d329a26.png)
![image](./images/6b95a44f34dbc29a663b78e3a05887a5.png)
![image](./images/1375f3072a6e41884d8d5482f49dc58c.png)
![image](./images/e4deb12ec91a9693305f735193753c32.png)
![image](./images/731107af71e760b0ffa31fadaf910748.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'breakup'
8. label:[0. 0. 0. 0. 1.]
![image](images/13f51f6f0a2111fea48031aa68bfd48c.png)
### Example 2 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_high'
8. label:[0. 1. 0. 0. 0.]
![image](images/2424b5138e19a4cb09c2a58b36f4f18d.png)
### Example 3 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_low'
8. label:[0. 0. 1. 0. 0.]
![image](images/f76009177d105a81db0c7c75fc7978c3.png)
### Example 4 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'nominal'
8. label:[1. 0. 0. 0. 0.]
![image](images/1dd6650be2a777d6d58ddcde6b567b3f.png)
### Example 5 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'rpo'
8. label:[0. 0. 0. 1. 0.]
![image](images/4fb66ac328e7a2c39333d0f0654c20ba.png)
