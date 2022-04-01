# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/45b60235a6a02a4b25d2dddd2bc20143.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/125fead938030e7f48f51a85acf958ca.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/b1a6c9f5ec011c68f3eaa004edb273f6.png)
![image](images/910464d0bae3fdc4580c9881119464eb.png)
![image](images/aa32c58a0f28100fc9f733737a4a052c.png)
![image](images/50f0e9fdb419047ad15273a4c47049c7.png)
![image](images/f53d255d1119123661ea7977b5325276.png)
![image](images/21c08421914132782905a9fd94eb26ce.png)
![image](images/5d673a8473979e4a04a18130d4b4f2e8.png)
![image](images/a09355237e7dfeacd9ca941b3a245c4d.png)
![image](images/d2c7e3344b356d12984e8ea4841637ea.png)
![image](images/5af1a5c6b495b90a60b695461e3e0ec3.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/97473ca3a80af9409a2da07384016155.png)
![image](./images/7b53336055127958a3c7f9515a6310f3.png)
![image](./images/7fb6ed199e8a9d126a534aec8fb4274e.png)
