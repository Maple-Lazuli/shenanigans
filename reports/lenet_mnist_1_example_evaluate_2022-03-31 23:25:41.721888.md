# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/713f8d2fa4febff8d45eec7ef9411bcc.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/308ad7f462b2496ab6aa9e4b73976817.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/e931a352d1a2a5b4b06ee3f0ffe690f0.png)
![image](images/fd41c7f3f98ba8ae16b4792367a71f83.png)
![image](images/c4d6bf00459f256e196ba317cd41627e.png)
![image](images/dd588f5753287f78dbb50adb8a9d5917.png)
![image](images/2a8b056e81fc9611631931f5d7172ed2.png)
![image](images/e0474ede162b224bafad169680e8b07a.png)
![image](images/c7d51234a6a24fe580038836ac7392dc.png)
![image](images/4a89e19969049a3494d91b3129e9e902.png)
![image](images/c0b2021da5c31452294ef0255e734111.png)
![image](images/49f87f349bfb5858024eccee5a3c9f8f.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf_1_train_example/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/ed70462bb9edea99f2543921205f31c0.png)
![image](./images/f41dd7cc39a2f8348287fead83253c7b.png)
![image](./images/cdc7e9804a41f0a86708b315c856e261.png)
