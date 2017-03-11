
# coding: utf-8

# In[3]:

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score


# In[22]:

# read in data
data = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", 
                     sep=",", na_values = ["?"])
data.columns = ["ID Number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
                "Normal Nucleoli", "Mitoses", "Class"]
# drop the "?"
data = data.dropna()
print(data.shape)

# testing set
data_cols = ["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
         "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
         "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

X = data[data_cols]
y = (data["Class"]
         .replace("2",0)
         .replace("4",1)
         .values.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = svm.SVC(kernel = "linear", gamma = 0.001, C = 100)
clf.fit(X_train, y_train)
# Prints the SVM coefficients
print (clf.coef_)

y_predict = clf.predict(X_test)



# Prints the precision, accuracy, and recall
print "Precision:  ", precision_score(y_test, y_predict)
print "Accuracy:  ", accuracy_score(y_test, y_predict)
print "Recall:  ", recall_score(y_test, y_predict)


# In[ ]:



