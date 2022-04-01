# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/43109d6d2ebd02f5046233fd5e8e7f0c.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/899f40ead69070b4ab256f3b19fe798d.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/c04301bb32ffefddf5410823c52a55c5.png)
![image](images/9d179ff4b0365289880a681650c0bd88.png)
![image](images/f0985e291301a326085722ff3a970041.png)
![image](images/f1db6435f0b6b09c127d88755cdfc235.png)
![image](images/17dd7ae6ba3d846207a0c7745d8638b3.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_tf_1_train_example/valid/satsim_valid.tfrecords consists of 383, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/ba699f68eda1d9eb690824f66b53de62.png)
![image](./images/7fea8f1dfc17783a1131753a2340ad3d.png)
![image](./images/9ba089a41d2ec69689c0ff478469f663.png)
