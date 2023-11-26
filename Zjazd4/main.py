"""
This program aims to calculate the classification accuracy of the given data.

For the program to work properly, you must provide formatted data on which it will work.

Authors: Dariusz Karasiewicz, Mikołaj Kusiński
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data_array_seeds = np.loadtxt('seeds_dataset.txt')
data_array_cancer = np.loadtxt('brest_cancer_dataset.txt')

X_cancer = data_array_cancer[:, 1:]
y_cancer = data_array_cancer[:, 0]
X_seeds = data_array_seeds[:, :-1]
y_seeds = data_array_seeds[:, -1]

def DecisionTreeAccuracy(x,y):
    """
    This function uses the NumPy and sklearn library to calculate the accuracy of data classification using a decision tree.
    First it creates decision tree model and use it to train it.
    Then it computes subset accuracy.

    Input:
        X: ndarray
        y: ndarray

    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=56)
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    y_pred_tree = clf_tree.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print(f'Decision Tree Accuracy: {accuracy_tree}')

def SVMAccuracy(x,y):
    """
    This function uses the NumPy and sklearn library to calculate the accuracy of data classification using a SVM.
    First it creates SVC model and use it to train it.
    Then it computes subset accuracy.

    Input:
        X: ndarray
        y: ndarray
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=56)
    clf_svm = SVC()
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f'SVM Accuracy: {accuracy_svm}')

print("Accuracy for wheat seeds data")
DecisionTreeAccuracy(X_seeds,y_seeds)
SVMAccuracy(X_seeds,y_seeds)
print("\n")
print("Accuracy for brest cancer data")
DecisionTreeAccuracy(X_cancer,y_cancer)
SVMAccuracy(X_cancer,y_cancer)