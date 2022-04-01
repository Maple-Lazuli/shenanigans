# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/812f9f65ea6e8ac3e0b2a5b41dd1f12e.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/553e3072d13c269616f09ab6371ff801.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/870943cad12898e2b2ab4881420f88cc.png)
![image](images/e56bf398637a823b55a864f2c8b96b16.png)
![image](images/430e73ea29107f2b3768afa2dac16bd2.png)
![image](images/c1751147fceeb2441a36396bc7209a7a.png)
![image](images/f8d1eb85acfb8043b2acaad8d469e795.png)
# Validation Dataset 
The validation dataset located at /media/ada/Internal Expansion/shenanigans_storage/generated_data_df/valid/satsim_valid.tfrecords consists of 353, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/cd4090816d83dc08c3d0ec2e4cb89a24.png)
![image](./images/f2a9e510cde96e0c7acafd24db287bc0.png)
![image](./images/83d61b77c296581d9f7c303a17792b52.png)
