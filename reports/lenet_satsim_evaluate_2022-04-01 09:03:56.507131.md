# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/c632f57fdde27a5137e1f01df4587c14.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/db2fa04a48ecd005c5b4466d420838fa.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/e32ed668b511b6e54cd142ba5b1c7790.png)
![image](images/e06bc3aefb6214391a5cf92489c92cea.png)
![image](images/dcbf1b4bf259a657a3d069cde9a6f0b1.png)
![image](images/51ecf64375a46ccbf09c57259915b53d.png)
![image](images/862ea6a5aa5eaea23fbc3b18ad0e1fbc.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 353, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/1a3075fd37fc88056b7c42bdea7c4f27.png)
![image](./images/4ba29ba00c814027a01799bf02b64227.png)
![image](./images/8ea549553921e7c2de7ccafbdebeba32.png)
