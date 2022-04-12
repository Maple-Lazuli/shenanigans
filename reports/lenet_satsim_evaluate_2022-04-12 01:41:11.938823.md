# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/43c311014db8a17fbfa86b3b277aebf1.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/63dbf31c898e00ed59a72e51b7042362.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/b8de1fc1ca838d3c128e051cce68e566.png)
![image](images/bb69ce438a5a9171c8443892c62f8202.png)
![image](images/eb81797a4e861065e4fb3018f6eee857.png)
![image](images/3b0722c39e18b121613ce29f2ab611b0.png)
![image](images/bc45624a07395d0208b0809b47b5ab71.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 353, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/1dbbbb31c58fb08fe1de1ac5eb6d7bab.png)
![image](./images/8f535ac89937a079c5ee5885320319ab.png)
![image](./images/f7363a8fcbd4cf7a8f3745c8d7dc9630.png)
