######## Spearman's Rank Correlation Function ###########################
######## Create a function that takes in x's and y's
import numpy as np
import pandas as pd
import scipy.stats

def my_fun_rank_corr (xdata, ydata):

    if (len(xdata)!= len(ydata)):
        return print ('Dimensions are not the same')
    else:
        x_rank=pd.Series(xdata).rank()
        y_rank=pd.Series(ydata).rank()
        n=len(xdata)
        k_x=len(pd.Series(xdata).value_counts())
        k_y=len(pd.Series(ydata).value_counts())
        if (n != k_x)|(n != k_y):
            return scipy.stats.pearsonr(x_rank, y_rank)[0]
        else:
            rs= 1 - ((6*sum((x_rank - y_rank)**2)) / (n*(n**2 -1)))
            return rs

x=[1,2,3,3,5,7]
y=[2,3,4,5,6,6]
# Run the function
a=my_fun_rank_corr(x, y)
print(a)        
#####################################################################
############## Decision Tree with Image ###########################
import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target

from sklearn.tree import DecisionTreeClassifier
DTclf=DecisionTreeClassifier()
DTclf.fit(df,y)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(DTclf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

##################### DT, RF & SVM #################################
from sklearn import datasets
iris=datasets.load_iris()

X=iris.data
y= iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=.3)

############### DT #################################################
from sklearn import tree
DTclf=tree.DecisionTreeClassifier()

DTclf.fit(X_train, y_train)
y_pred= DTclf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

################# RF ############################################
from sklearn.ensemble import RandomForestClassifier
RFclf=RandomForestClassifier(n_estimators=11)

RFclf.fit(X_train, y_train)
y_pred=RFclf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

####################### SVM ######################################
from sklearn.svm import SVC
SVclf = SVC(kernel='poly', degree=4)
#### kernel='linear', Gaussian kernel: kernel = 'rbf', kernel='sigmoid'

SVclf.fit(X_train, y_train)
y_pred=SVclf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

################################################################################
################################################################################
####################### ROC for Binary Class ###################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris=datasets.load_iris()

X=iris.data
y= iris.target
# Taking two classes of iris flowers
X=X[0:99,:]
y=y[0:99]

# shuffle and split training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)

################### DT: ROC for Binary Class ###########################################
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
###Predict probabilities for the test data.
probs = clf.predict_proba(X_test)
####Keep Probabilities of the positive class only.
probs = probs[:, 1]

from sklearn.metrics import roc_curve, roc_auc_score
###Compute the AUC Score.
auc = roc_auc_score(y_test, probs)
print('AUC:', auc)
###Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
####Plot ROC Curve 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

####################### RF: ROC for Binary Class #####################################
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=11)
clf.fit(X_train, y_train)
###Predict probabilities for the test data.
probs = clf.predict_proba(X_test)
####Keep Probabilities of the positive class only.
probs = probs[:, 1]

from sklearn.metrics import roc_curve, roc_auc_score
###Compute the AUC Score.
auc = roc_auc_score(y_test, probs)
print('AUC:', auc)
###Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
####Plot ROC Curve 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

######################### SVM: ROC for Binary Class ##################################
from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=4)
probs = clf.fit(X_train, y_train).decision_function(X_test)

from sklearn.metrics import roc_curve, roc_auc_score
###Compute the AUC Score.
auc = roc_auc_score(y_test, probs)
print('AUC:', auc)
###Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
####Plot ROC Curve 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

############################################################################################
################################### K-fold ################################################
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
