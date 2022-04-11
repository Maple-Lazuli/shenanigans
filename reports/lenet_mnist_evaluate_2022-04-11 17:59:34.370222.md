# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/a6439f9493a7256d706d6793bf667a33.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/c0045241f8c17f2345e01d59b498b2cb.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/881afef32a2e0a57865c210c8ebd6e7c.png)
![image](images/1a51f5abe508b0e94511520d0030b18d.png)
![image](images/f0bc69d6e55763a4c4c3cf5fa8198149.png)
![image](images/adacac1edf09c03904013adc8a35169e.png)
![image](images/f37662281357cdb73d1c09a0af31378e.png)
![image](images/c75263a8580ee609fc3e0266b16be740.png)
![image](images/9b5925048b9677830a3a3fb5f9ac5339.png)
![image](images/aae50f9da9c1f9da1ca1d26176021848.png)
![image](images/a0ff89775e41c2cbf37b0f6b20fbcadd.png)
![image](images/5d9c7627847e43f2dff61781ccd170f7.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/afef258e612d702e9b80e1f90814a150.png)
![image](./images/11e3fa67dca6abf68708dfab9939ffa1.png)
![image](./images/47ca0f5abca3af19e14c670f275e7ac2.png)
