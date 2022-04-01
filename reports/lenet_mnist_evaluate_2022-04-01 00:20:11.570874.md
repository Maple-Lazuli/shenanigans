# Confusion Matrix Against Validation Set
The multi-class confusion matrix captures the true labels along the columns and the predicted labels along the rows. The cells contain counts for the intersection of true labels and predicted labels. 
![image](images/5767a41f5cea54f776b955d451e51b9e.png)
# Score Matrix 
The score matrix contains the true positives, the true negatives, the false positives, the false negatives, the precision, the recall, the specificity, the misclassification rate, accuracy, and the f1 score for each labelthe classifier is trained on. 
![image](images/1d93832cecd37b084fa39df95388b530.png)
# Multiclass Receiver Operating Characteristic (ROC) Curves 
The multiclass ROC curves were created using one-versus-rest classifications against the validation set.
![image](images/52f7b9963590e4b8c69128cf97cbb6aa.png)
![image](images/9265b121a2c30b65dbb72c4d0ba8484d.png)
![image](images/3ea2987e33546b287e75e06f020d3ed4.png)
![image](images/4d93e05f622e1c94747bebfe389e47a4.png)
![image](images/e31528a13e415ff391cae6c36197be10.png)
![image](images/07a6f1d37ce14ba410916e3ca8d3ee6b.png)
![image](images/68d401b2e21331ff080878bcf163fd91.png)
![image](images/99a0cbb756c7e35d8aa96e091e2f4997.png)
![image](images/8bb6e4a008dac73a2b303cbceefce8b8.png)
![image](images/d202d833b0c1977c35375c54138e458a.png)
# Validation Dataset 
The validation dataset located at ./mnist_tf/valid/mnist_valid.tfrecords consists of 14000, served in batch sizes of 1.
 The charts below depict the distribution of the features of this dataset![image](./images/e43e2dfbfecc3afaa5fd294c6f674baf.png)
![image](./images/11063cff7f458580e6d89ac26b978027.png)
![image](./images/59be1a84bc4f5383465874f4f06bc6e1.png)
