print 'Importing Required Modules'
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
import glob
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter

#This points to our most volatile stock price data file
file = 'NFLX_data.csv'

#The threshold value is the over/under point that separates
#  our two classes
threshold = float(275)
cv = 10

#From 11/19/2013
#test_data = ['339.49', '344.37', '332.79','2961900','335.63']
test_data = ['339.49','344.37','332.79','2961900']

def loadData(file):
    data = open(file).read()
    return data

#The organizeData function builds our input data for
#  sklearn from our scraped data file

def organizeData(data,threshold):
    data = data.split('\n')
    X = []
    y = []
    # The syntax data[1:-1] removes the first (zeroth)
    #   and last lines of the file
    for d in data[1:-1]:
        tmp=[]
        d = d.split(',')
#        tmp.append(d[0]) # Date [irrelevant - not a time series]
        tmp.append(d[1]) # Open
        tmp.append(d[2]) # High
        tmp.append(d[3]) # Low
#        tmp.append(d[4]) # Close [can't use this - cheating]
        tmp.append(d[5]) # Volume
#        tmp.append(d[6]) # Adjusted Close [also cheating]
        X.append(tmp)
 
       #Generate a vector Y with our class labels
        if float(d[6])>=threshold:
            y.append(1)
        elif float(d[6])<threshold:
            y.append(0)
    return np.array(X),np.array(y)

def classify(X,y,cv):
    #clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=10,random_state=5)
    #clf = RandomForestClassifier(n_estimators=1000)
    clf = AdaBoostClassifier()
    #clf = ExtraTreesClassifier()

    score = cross_val_score(clf, X, y, cv=cv)
    print '%s-fold cross validation accuracy: %s' % (cv,sum(score)/score.shape[0])
    clf = clf.fit(X,y)

    #print 'Feature Importances'
    #print clf.feature_importances_
    #X = clf.transform(X,threshold=.3)
    
    preds = clf.predict(X)
    print 'predictions counter'
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
    
    print 'correct positives:', tp
    print 'correct negatives:', tn
    print 'false positives:', fp
    print 'false negatives:', fn
    print 'precision:',float(tp)/(tp+fp)
    print 'recall (tp)/(tp+fn):',float(tp)/(tp+fn)
    print 'false positive rate (fp)/(fp+tn):', float(fp)/(fp+tn)
    print 'false positive rate2 (fp)/(fp+tp):', float(fp)/(fp+tp)
    print 'prediction accuracy: %s%s\n' % (100*float(tp+tn)/(tp+tn+fp+fn),'%') 
    return clf

def testOOS(clf,test_data,threshold):
    #Make sure you know the difference between the predict() and
    # predict_proba() methods. Hint: What does each return?
    pred_proba = clf.predict_proba(test_data)
    pred = clf.predict(test_data)

    if float(pred[0])==float(1):
        print '***Classified as likely to close higher than $%s per share' % threshold
    elif float(pred[0])==float(0):
        print '***Classified as likely to close lower than $%s per share' % threshold
    print 'Probability of closing higher than $%s per share: %s' % (threshold,pred_proba[0][1])
    print 'Probability of closing lower than $%s per share: %s' % (threshold,pred_proba[0][0])

#Main body of script
#
print 'loading data'
data = loadData(file)
print 'organizing data'
X,y = organizeData(data,threshold)
print 'training'
clf=classify(X,y,cv)
print 'testing oos data (open, high, low, volume): %s' % test_data
testOOS(clf,test_data,threshold)
print '\nDONE!!!'
