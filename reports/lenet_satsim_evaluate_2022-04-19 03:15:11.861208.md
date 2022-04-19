# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/63e8b134c49b59ef1a30b7e7ff36edfa.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/ddf817702a7a0742b46a5155d7b00bc3.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/acdc301c3a7687f1b506f6c907bc51fc.png)
![image](images/6d22393a3d849564d03260c046605df9.png)
![image](images/3a82bc1430511600003a1b80b804284a.png)
![image](images/ee3c58b09702f7b64e84f0ec3259e6b6.png)
![image](images/cebbce379c7dee26738d5f5ece197a83.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 950, served in batch sizes of 50.
 The charts below depict the distribution of the features of this dataset![image](./images/a4d63567dbefd7ab84e734c9e144662b.png)
![image](./images/de61a762403ae1ae3f9bfb73a8c67307.png)
![image](./images/9fe0feacf31ced40f680a03297a16242.png)
