# Overview 

        Implementation of a variant of lenet to classify simulated satellite events observed through electro-optical imagry. 
        
# Hyperparameters 
This section documents the hyperparameters used for this session. 
1. learning_rate: 0.001
2. epochs: 1
# Performance
![image](images/81b93650b004f0da1c9b2ee4ad3b5ba2.png)
![image](images/44f033baea527516be92838926de81fa.png)
![image](images/04a0337390ec225649498885bbea1c54.png)
# Datasets 
### Training Set 
The training set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/train/satsim_train.tfrecords consists of 550, served in batch sizes of 50.

### Validation Set 
The validation set located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 950, served in batch sizes of 50.

### Validation Set and Training Set Comparison 
This section compares the contents of the validation and train sets used.
![image](./images/25de6fa299313d2e4a671cf377fabf49.png)
![image](./images/fa983630415a92d39ecba5890c8ee7c4.png)
![image](./images/228c5c3949d18b7af78b9bf8fe54d700.png)
![image](./images/784208982afa7587271a8060ad6ce00e.png)
![image](./images/7be048ad7b2b3a218a7601257be5d6b0.png)
![image](./images/98f672f3b5e82d0f0ad3131f7d10ed84.png)
![image](./images/dbfec1823cc5aab296b8ec007a2bb402.png)
![image](./images/253413b0bb3421463e54b06801364975.png)
![image](./images/445389903b38c3adc73a5ece3ca3d12c.png)
![image](./images/5fd0796239ada9dace532d2d2cadb600.png)
![image](./images/c5e5dd92844d92a04187d7dc089198bc.png)
![image](./images/f31cef963fdf2117ff235e1c9de1508b.png)
# Dataset Examples
This section depicts one input for each label the model is expected to learn.
### Example 1 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_low'
8. label:[0. 0. 1. 0. 0.]
![image](images/7eda93ba93cdfb4e095649b14e771421.png)
### Example 2 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'breakup'
8. label:[0. 0. 0. 0. 1.]
![image](images/a07d139f0ee27de175d5f0fa2e792c38.png)
### Example 3 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'collision_high'
8. label:[0. 1. 0. 0. 0.]
![image](images/cbf8580fc47f5b501844f79ff3374f2a.png)
### Example 4 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'nominal'
8. label:[1. 0. 0. 0. 0.]
![image](images/0c656fee04cae5ae01a3cda1ce907ad2.png)
### Example 5 
1. height:512
2. width:512
3. depth:16
4. field_of_view_x:0.0006727635045535862
5. field_of_view_y:0.0006727635045535862
6. stray_light:0
7. class_name:b'rpo'
8. label:[0. 0. 0. 1. 0.]
![image](images/7c43dd80859edeca9618f5472ff4cb6f.png)
