from sklearn import datasets
iris=datasets.load_iris()

X=iris.data
y= iris.target

DTscores = []
RFscores = []
SVscores = []
############### DT ###########################################
from sklearn import tree
DTclf=tree.DecisionTreeClassifier()

############### RF ############################################
from sklearn.ensemble import RandomForestClassifier
RFclf=RandomForestClassifier(n_estimators=11)

######################### SVM ##################################
from sklearn.svm import SVC
SVclf = SVC(kernel='poly', degree=4)

from sklearn.model_selection import KFold
cv = KFold(n_splits=3, random_state=1, shuffle=True)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    ############# for DT #################
    DTclf.fit(X_train, y_train)
    DTscores.append(DTclf.score(X_test, y_test))
    ############# for RF #################
    RFclf.fit(X_train, y_train)
    RFscores.append(RFclf.score(X_test, y_test))
    ############# For SVM #################
    SVclf.fit(X_train, y_train)
    SVscores.append(SVclf.score(X_test, y_test))
    


import numpy as np
print('DTscores:', np.mean(DTscores))
print('RFscores:', np.mean(RFscores))
print('SVscores:', np.mean(SVscores))
