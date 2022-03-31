# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/9b752824969b15ae355d11de40a5284f.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/326ee09464d33b2bf680307359053783.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/56cf0b0756d2698b93a73b284463450f.png)
![image](images/42d86d323725317e1061ffcf825b65db.png)
![image](images/5e779e79bfb74bbb08cc2b4f2fb322ec.png)
![image](images/9d919e5a9d4eea7a1f9521f2a24b7de7.png)
![image](images/d0b8bd0b97b723bb2d667232903d9de7.png)
![image](images/2a97ce1fa0247ebc1f26eeb02ad93473.png)
![image](images/33938482e53e1af77c6979f0211b3b1b.png)
![image](images/6838072d0e8b115f09678eb3cd0bdca0.png)
![image](images/694e499a6b8d196c9c93cf2ada00afe6.png)
![image](images/7639e4b8d8c0aac9fc721b49dae68c26.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 7000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/2a936345d989207cf87aac892f8f82cd.png)
![image](./images/e7b2629c2f59623ea0aefb24e80c214b.png)
![image](./images/9fbe145b29480ab4e2ef6a2fb362d522.png)
