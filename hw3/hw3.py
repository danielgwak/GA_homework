print 'Importing Required Modules...'
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
import glob
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
print 'Module Imports Complete.'

#This points to our most volatile stock price data file
file = 'BTH_data.csv'

#The threshold value is the over/under point that separates
#  our two classes
threshold = float(14)
cv = 10

# Load the file into memory. 
def loadData(file):
    data = open(file).read()
    return data

# Build input data for the classifier
def organizeData(data,threshold):
    data = data.split('\n')
    X = []
    y = []

# Below, I have decided to only use Open Price and Trading volume. Giving the Close Prices is cheating by training the answer into the algorithm.  I think giving the High and Low Prices also makes the challenge trivial, so I am excluding them.
    for d in data[1:-1]:
        tmp=[]
        d = d.split(',')
#        tmp.append(d[0]) # Date [irrelevant - not a time series]
        tmp.append(d[1]) # Open
#        tmp.append(d[2]) # High
#        tmp.append(d[3]) # Low
#        tmp.append(d[4]) # Close [can't use this - cheating]
        tmp.append(d[5]) # Volume
#        tmp.append(d[6]) # Adjusted Close [also cheating]
        X.append(tmp)
        # print tmp
 
       #Generate a vector Y with our class labels
        if float(d[6])>=threshold:
            y.append(1)
        elif float(d[6])<threshold:
            y.append(0)
    return np.array(X),np.array(y)

def classify(X,y,cv):
    clf = AdaBoostClassifier()

    score = cross_val_score(clf, X, y, cv=cv)
    print '%s-fold cross validation accuracy: %s' % (cv,sum(score)/score.shape[0])
    clf = clf.fit(X,y)
    
    preds = clf.predict(X)

    # The below measures are from Rob's GitHub code
    print 'Predictions Counter'
    print Counter(clf.predict(X))
    fp=0
    tp=0
    fn=0
    tn=0
    for a in range(len(y)):
        if y[a]==preds[a]:
            if preds[a]==0:
                tn+=1
            elif preds[a]==1:
                tp+=1
        elif preds[a]==1:fp+=1
        elif preds[a]==0:fn+=1
    
    print 'True Positives:', tp
    print 'True Negatives:', tn
    print 'False Positives:', fp
    print 'False Negatives:', fn
    print 'Precision:',float(tp)/(tp+fp)
    print 'Recall (tp)/(tp+fn):',float(tp)/(tp+fn)
    print 'False Positive Rate (fp)/(fp+tn):', float(fp)/(fp+tn)
    print 'False Positive Rate2 (fp)/(fp+tp):', float(fp)/(fp+tp)
    print 'Prediction Accuracy: %s%s' % (100*float(tp+tn)/(tp+tn+fp+fn),'%') 
    return clf


# Running the above components

print 'Loading Data...'
data = loadData(file)
print 'Loading Data Complete.'

print 'Organizing Data for Training'
X,y = organizeData(data,threshold)
print 'Data Organization Complete.'

print 'Training Classifier...'
clf=classify(X,y,cv)
print 'Training Complete.'


# Evaluate Classifier Using ROC

fpr, tpr, thresholds = roc_curve(y, clf.predict(X))
roc_auc = auc(fpr, tpr)
print 'Area under the ROC curve: %s' % roc_auc
