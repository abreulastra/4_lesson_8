# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:33:41 2016

@author: AbreuLastra_Work
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values

# Split the data X and y in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.4 )


# search for an optimal value of C for SVM
k_range = list(range(1, 10))
k_scores = []
for k in k_range:
    svc = svm.SVC(kernel='linear' , C=k)
    scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())



# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of C for SVM')
plt.ylabel('Cross-Validated Accuracy')

# check classification accuracy of SVM

svc = svm.SVC(kernel='linear', C=1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))