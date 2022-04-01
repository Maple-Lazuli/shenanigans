# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 30
# Performance
![image](images/ba97ea2e5748d087811f2ebf6fdbc047.png)
![image](images/6e2ae1ed09a38954c4b448c05bf45cdb.png)
![image](images/d83cc953a2a65a734f6bdbe0aa481239.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/train/satsim_train.tfrecords consists of 1200, served in batch sizes of 50.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 350, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/706ddcb6a5857d190792118892b6a409.png)
![image](./images/fd2457ef6b8813279fac9f92dafd0bd1.png)
![image](./images/a70f1928444ba91f74baed4e07f44d88.png)
![image](./images/3d4654bf1b1eac129bd8bb222ac90662.png)
![image](./images/b52791519402a682405181550bde48c4.png)
![image](./images/394ac66be59d97741a29fcca169faff8.png)
![image](./images/a227b76b793d953844cc6b7742868474.png)
![image](./images/699862378ce206fbb0bc04e468cd2bec.png)
![image](./images/2d09e26d1fa6423f87c487423adbb4d6.png)
![image](./images/48d02b1132d33d8f2c03f608f422ee6f.png)
![image](./images/ad35da939cac666f3fca7bf126722568.png)
![image](./images/f6171a17a3af64539a4f509586796681.png)
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
![image](images/dad95333ec744010532719c15d6dd6c6.png)
### Example 2 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_high'
8. label:[0. 1. 0. 0. 0.]
![image](images/1523836deba5afc555392f63a88c3157.png)
### Example 3 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_low'
8. label:[0. 0. 1. 0. 0.]
![image](images/3885a421f134b57ed46ce7638f6e75d2.png)
### Example 4 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'rpo'
8. label:[0. 0. 0. 1. 0.]
![image](images/b309d0c48adfd2b13e2a7cc9a5619a64.png)
### Example 5 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'nominal'
8. label:[1. 0. 0. 0. 0.]
![image](images/a92a2fa5998ff128d74019a78400cc3a.png)
