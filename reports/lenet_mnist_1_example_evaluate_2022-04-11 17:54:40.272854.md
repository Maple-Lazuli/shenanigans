# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/9039f9687c555729152ae73969271439.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/e37edb7a13121b1639e67ddd1582aa4f.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/7f1ee01754b1687239824134ad032975.png)
![image](images/9d6659a711e64a9e64c49d766773d692.png)
![image](images/e0f9289ec380c83b5531761cc749205c.png)
![image](images/4ae877b047b60497532ab3d62c0055f1.png)
![image](images/f960dc35e6e7868227f76a96553f01dc.png)
![image](images/23fded4091dd88c9a0dbc9dd0e3b456e.png)
![image](images/08b4044550aa03d76f3a53d7dcdb801c.png)
![image](images/833522ad5d3d918ba6377e62794e7331.png)
![image](images/68db5e8de28dadef787aad16c4014c37.png)
![image](images/8bc827189b82f9ef2cf7c256bc7b9f52.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/c1cc886c44632f133a7917a128a5deec.png)
![image](./images/9d6253cdb7a28b49d042cbf57d8c6ae6.png)
![image](./images/5b170176a01ea3fb61c3d20cb0d22a12.png)
