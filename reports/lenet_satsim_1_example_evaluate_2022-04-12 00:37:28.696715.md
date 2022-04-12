# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/1070dc2f4d54b8430525def72794cfd5.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/36ad195588acd079b68270e1372438a1.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/a4632687e9335afd437988a113280c66.png)
![image](images/dd5eddc38a391ed30180ce28b4f29065.png)
![image](images/fec2ec5977174db853044e399811b4c2.png)
![image](images/ed7c1cd3836c421f192a95a05b09e44f.png)
![image](images/4f16a61ff7a0e421ec07f563a7e377b9.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/valid/satsim_valid.tfrecords consists of 383, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/5e6873a8be3032410d9c69d6073e1e93.png)
![image](./images/e6e33e4a5e3f76380b468d48446c75a9.png)
![image](./images/4a1ed0d93112a6e11a92b6f21675be44.png)
