# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/cc3b93c65e370c44c78a3c1750bf92c4.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/2703a9b85afedaa0a7d49e9115ce0fef.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/c30151388fbd71eb74edd6ca3d7842d7.png)
![image](images/797d86562f39fd6174a4567181ac3742.png)
![image](images/30fa4bb00fe6ac3624b5bfd9e894d5fe.png)
![image](images/2cb8700e782dd1fab041da555af12c92.png)
![image](images/e3903d5deaceba562f61a1f804af215e.png)
![image](images/54422a4721d388cf1287da473289cf99.png)
![image](images/12455e314424b8956621b0df088eb0fb.png)
![image](images/fa28e361d55ddf46d741bbfc7b26ac16.png)
![image](images/7a7f0b8588b4dda61a4214a179a627ab.png)
![image](images/f4f064cab3be45035c01ab90a603c630.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/dc70c6a9baf8bf20446e11de19ce95c3.png)
![image](./images/bce3341ab50a9bc6195fd0245960711d.png)
![image](./images/028a307c785e9df795bd788adad74843.png)
