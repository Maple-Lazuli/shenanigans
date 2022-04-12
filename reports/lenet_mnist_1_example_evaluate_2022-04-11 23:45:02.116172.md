# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/a6e70288fb635b62b13ca9b96fa1758f.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/3348392b5273d5969f709ffc8d928d41.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/a5ce29cd05ae81b7a7ded9b621f2b39f.png)
![image](images/53b8203d59f0e94c353758d325b3ab38.png)
![image](images/dabbbe7dbc22a3e14e3e1a8f847b81ae.png)
![image](images/e6602d78c1cf05b6a26f885b9a556dda.png)
![image](images/411713176f9cf41a22a34c6105fbeb21.png)
![image](images/75cce891f46bf6ec72f332718a95b9a4.png)
![image](images/578b2974e928895c36f0402db9149ae4.png)
![image](images/50e84bdb45f67e8027d871b6181879d0.png)
![image](images/7cb9983d53998da157702662a397dd45.png)
![image](images/d23d66e3cc782f9eba73c48abff92ad4.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/743479155d53c81354b736453f803b96.png)
![image](./images/3421ca33c6e61248d3b6adcad860bb62.png)
![image](./images/0a1bf7e25a1a7732be0ab47b15160e1d.png)
