import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd

seed = 10
np.random.seed(seed)

X = np.load('enron_features_matrix.npy');
y = np.load('enron_labels.npy');

models = {
        "XGBClassifier": XGBClassifier(),
        "LinearSVC": LinearSVC(max_iter=1000000),
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(C=2.7825594022071245, penalty = 'l2'),
        "RandomForest": RandomForestClassifier(),
        }

final_results = []

for model_name in models:
    accuracy = []
    sensitivity = []
    specificity = []
    precision = []
    f1_score = []
    mcc = []
    skf = StratifiedKFold(n_splits=10)
    for train_index, val_index in skf.split(X, y):
        print("Train:", train_index, "Validation:", val_index)
        X_train, X_test = X[train_index], X[val_index]
        y_train, y_test = y[train_index], y[val_index]
        model = models[model_name]
        model.fit(X_train,y_train)
        result = model.predict(X_test)
        c_matrix = confusion_matrix(y_test, result)
        TN = float(c_matrix[0][0]) #Correcly predicted not spam
        FP = float(c_matrix[0][1]) #Not spam, but incorrectly predicted as spam
        FN = float(c_matrix[1][0]) #Spam, but incorrectly predicted as not spam
        TP = float(c_matrix[1][1]) #Correctly predicted spam
        accuracy.append((TP+TN)/(TP+FN+TN+FP))
        sensitivity.append(TP/(TP+FN))
        specificity.append(TN/(TN+FP))
        precision.append(TP/(TP+FP))
        f1_score.append(2*TP/(2*TP+FP+FN))
        mcc.append(((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5))
    final_results.append([])
    final_results[-1].append(sum(accuracy)/len(accuracy))
    final_results[-1].append(sum(sensitivity)/len(accuracy))
    final_results[-1].append(sum(specificity)/len(specificity))
    final_results[-1].append(sum(f1_score)/len(f1_score))
    final_results[-1].append(sum(mcc)/len(mcc))

columns = ["Accuracy", "Sensitivity", "Specificity", "F1_score", "Mcc"]
index = [i for i in models]
df = pd.DataFrame(data=final_results, columns=columns, index=index)
df.to_csv("spam_model_results.csv")

    
