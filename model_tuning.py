import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import pandas as pd

seed = 10
np.random.seed(seed)

X = np.load('enron_features_matrix.npy');
y = np.load('enron_labels.npy');

model = LogisticRegression(C=2.7825594022071245, penalty = 'l2')
#penalty = ['l1', 'l2']
#C = np.logspace(0, 4, 10)
#hyperparameters = dict(C=C, penalty=penalty)
#clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)
#best_model = clf.fit(X, y)
#print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
#print('Best C:', best_model.best_estimator_.get_params()['C'])

y_predict = best_model.predict(X)
c_matrix = confusion_matrix(y, y_predict)
TN = float(c_matrix[0][0]) #Correcly predicted not spam
FP = float(c_matrix[0][1]) #Not spam, but incorrectly predicted as spam
FN = float(c_matrix[1][0]) #Spam, but incorrectly predicted as not spam
TP = float(c_matrix[1][1]) #Correctly predicted spam
print("accuracy: " + str((TP+TN)/(TP+FN+TN+FP)))
print("sensitivity: " + str(TP/(TP+FN)))
print("specificity: " + str(TN/(TN+FP)))
print("precision: " + str(TP/(TP+FP)))
print("f1_score: " + str(2*TP/(2*TP+FP+FN)))
print("mcc: " + str(((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)))


    
