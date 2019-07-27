# Email Spam Classifier

This repository contains a project for classifying spam. Since this is a school project, a detailed write-up can be found in WriteUp.docx.

Brief summary of scripts:

*spam_preprocess.py* - Pre-process the data and labels for classification.

*spam_model.py* - Experiment which different baseline classification models.

*model_tuning.py* - Grid search (not extensive due to lack of time) for parameter optimization.

To replicate the results:

1.  Download Enron-data-set and make sure it is in the same directory as the scripts.
2. Remove all readmes in Enron-data-set and the only folder structure should be:
   Enron-data-set
   	Enron1
   		ham
   		spam
   	Enron2
   		ham
   		spam
   	Enron3
   		ham
   		spam
   	Enron4
   		ham
   		spam
   	Enron5
   		ham
   		spam
   	Enron6
   		ham
   		spam
   NOTE: ALL MUST BE FOLDERS!! The mails should only be found in ham and spam folders.