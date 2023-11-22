import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# data_array = np.loadtxt('seeds_dataset.txt')
data_array = np.loadtxt('brest_cancer_dataset.txt')

X = data_array[:, :-1]
y = data_array[:, -1]

def DecisionTreeAccuracy(x,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    y_pred_tree = clf_tree.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print(f'Dokładność Drzewa Decyzyjnego: {accuracy_tree}')

def SVMAccuracy(x,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf_svm = SVC()
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f'Dokładność SVM: {accuracy_svm}')

DecisionTreeAccuracy(X,y)
SVMAccuracy(X,y)
