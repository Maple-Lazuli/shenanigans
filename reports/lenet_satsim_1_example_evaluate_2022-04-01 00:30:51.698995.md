# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/fd1f13fd89089c5984a97db8d93c97f2.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/8bde77ac5babd922612cb04b4f4a0d06.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/9a7bd744a3f18bdcacab11d349657e0e.png)
![image](images/1b56a6ba3a3616483a7958ce0ab46fd6.png)
![image](images/0431f8f6d8b823de55367f4015ff5b89.png)
![image](images/17b133d8e5cfe78a7af6d8e134bb4bfa.png)
![image](images/32829afb745cd961edfa54bc8e13a3fc.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/valid/satsim_valid.tfrecords consists of 383, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/27daf3ac1d154586010c8fd74f3ce98a.png)
![image](./images/f2d3f2c49ecde71b01f607497ae503e3.png)
![image](./images/e204b6adadde0f04c94f472c8cdede31.png)
