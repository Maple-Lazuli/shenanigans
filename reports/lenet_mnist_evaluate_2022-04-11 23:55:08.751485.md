# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/fadf723d20a0a19150fe75fa66c3d6f0.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/500bae28ea810dcc105f036e856ac24e.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/52d9b8bca9b28b6759c2e89d72a4a3cf.png)
![image](images/86022c9c2b0921273ab772f900fd6971.png)
![image](images/81c2cab926e76be71dfaafd0bac33dd7.png)
![image](images/1dc477c4af032f3739e31d46efa8e04d.png)
![image](images/813801932b4a3339310c0620ad9514ad.png)
![image](images/208348519743a9676c543a5909cc17a5.png)
![image](images/6ff83e37c9941f127dd2533923f5d471.png)
![image](images/02d7e51a83022270e30ce036ca26b031.png)
![image](images/4ec525a9e2646dd0ac800bbde8a9703c.png)
![image](images/b7bb864187a562dbe67e3f4944c979d1.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/78f95bca1db06116d3cbc2a778a110ff.png)
![image](./images/44dc0d6d2271b5475c04c9a96b0a08b6.png)
![image](./images/d65f137bde5ea261c19ad5aa8ee4980e.png)
