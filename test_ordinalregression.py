""" Training script for classification based on segmented pixel count
Tutorial from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://www.datacamp.com/tutorial/understanding-logistic-regression-python"""

import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


Xl = pd.read_excel("C:/Netwerken/FromScratch/Ordinal_regression/Pixel_score_data.xlsx")
Xl = pd.DataFrame.to_numpy(Xl)
X = Xl[:,1]
X = X.reshape(1,-1)
X = X.astype(float)
X = X.transpose()
y = Xl[:,2]
y = y.astype(float)

# split X and y into training and testing sets
X_train,X_test,y_train,y_test=  train_test_split(X,y,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

# import the metrics class
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1,2,3] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, fmt="d")
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Score 0', 'Score 1', 'Score 2', 'Score 3'])
ax.yaxis.set_ticklabels(['Score 0', 'Score 1', 'Score 2', 'Score 3'])
plt.tight_layout()
plt.show()

# Confusion matrix per class
mcm = multilabel_confusion_matrix(y_test, y_pred)
mcm_score0 = mcm[0]
mcm_score1 = mcm[1]
mcm_score2 = mcm[2]
mcm_score3 = mcm[3]

# Plots
ax = sns.heatmap(mcm_score0, annot=True)
ax.set_title('Confusion Matrix Score 0')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 0'])
ax.yaxis.set_ticklabels(['Else','Score 0'])
plt.show()

ax = sns.heatmap(mcm_score1, annot=True)
ax.set_title('Confusion Matrix Score 1')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 1'])
ax.yaxis.set_ticklabels(['Else','Score 1'])
plt.show()

ax = sns.heatmap(mcm_score2, annot=True)
ax.set_title('Confusion Matrix Score 2')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 2'])
ax.yaxis.set_ticklabels(['Else','Score 2'])
plt.show()

ax = sns.heatmap(mcm_score3, annot=True)
ax.set_title('Confusion Matrix Score 3')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 3'])
ax.yaxis.set_ticklabels(['Else','Score 3'])
plt.show()

count0 = np.count_nonzero(Xl[:,2] == 0)
count1 = np.count_nonzero(Xl[:,2] == 1)
count2 = np.count_nonzero(Xl[:,2] == 2)
count3 = np.count_nonzero(Xl[:,2] == 3)
frequency = [count0, count1, count2, count3]

classes = ["Score 0", "Score 1", "score 2", "Score 3"]

# creating the bar plot
plt.bar(classes, frequency, color='maroon',
        width=0.4)
plt.xlabel("LUS classes")
plt.ylabel("Number of occurences")
plt.show()
