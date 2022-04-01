# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 20
# Performance
![image](images/85d6fe6c4f578289cb7ca8cf0de3dc6b.png)
![image](images/f13cbfbe71095041a15b7df2f5c2bdd3.png)
![image](images/6034ce7ad1b47635cf25a0e9a15db877.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/train/satsim_train.tfrecords consists of 1200, served in batch sizes of 50.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 350, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/3164933d0a87a85a3f85281914c9ceda.png)
![image](./images/073086813cd72c89931cee688658423d.png)
![image](./images/dfc1c10fd0e574f6cadc0f9520e6a9f2.png)
![image](./images/a9062ff83af2ae1383f6f9594742ec53.png)
![image](./images/28fcfa5d464542de7d302abc41d14521.png)
![image](./images/66ffa593a183241b70353a4b20cbb8ec.png)
![image](./images/16160b7cc0d36926265ce80dbf0d9c56.png)
![image](./images/09f50b685b4f14e1ecf454eea4a9747a.png)
![image](./images/2e90f95f7e842039fdaf8db78434b42b.png)
![image](./images/3b28a02dc37cc821ff18487c45871f1c.png)
![image](./images/369ccd61c623f9ded3ee10b00438fc18.png)
![image](./images/85eb55fc931fd117e963c0798def1a36.png)
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
![image](images/1cb77c382047a285b74cb445706210b9.png)
### Example 2 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_high'
8. label:[0. 1. 0. 0. 0.]
![image](images/068c20e59d8737109e90f077aa32993b.png)
### Example 3 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_low'
8. label:[0. 0. 1. 0. 0.]
![image](images/b3475240aa61b0bc353e889800369a66.png)
### Example 4 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'nominal'
8. label:[1. 0. 0. 0. 0.]
![image](images/b51d201a7e3867ca72599e481b8169cd.png)
### Example 5 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'rpo'
8. label:[0. 0. 0. 1. 0.]
![image](images/2f2e0e69742806393d1202f6939421e8.png)
