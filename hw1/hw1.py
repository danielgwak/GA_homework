<<<<<<< HEAD
# This section defines a function that imports the iris dataset

from sklearn import datasets

def load_iris_data():
    iris = datasets.load_iris()
    return (iris.data, iris.target, iris.target_names)

# This section sets up a kNN function with k-fold crossvalidation

from sklearn.neighbors import KNeighborsClassifier

def knn(x_train, y_train, k_neighbors=3):
    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(x_train, y_train)
    return clf

from sklearn.cross_validation import KFold

def cross_validate(XX,yy,classifier,k_fold):
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True)
    k_score_total = 0

    for train_slice, test_slice in k_fold_indices:
        model = classifier(XX[[ train_slice ]],
                           yy[[ train_slice ]])

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    return k_score_total*1.0/k_fold

""" 
Execute HW1 below, based on functions defined above
"""


k_range = range(2,100)  #Set the range of K-fold lengths

classifier = knn #Define the classifier used

(XX,yy,y) = load_iris_data() #Load the iris data into a list

results = []  #Prepare an empty list called 'results' that will contain the outputs

for i in k_range:
#Iterate through all the k-folds in our k_range list
    results.append([i,cross_validate(XX,yy,classifier,i)])
#Append the results from that analysis to our 'results' list

ranked_results = sorted(results, key=lambda x: x[1], reverse = True)
#re-order the 'results' list from max to min by the cross-validation score

print ranked_results
#Print and take a look at which number of folds gives us the highest score

"""
Answers to written portion of HW1 are below:

1) Problem we are trying to solve --
    The problem we are trying to solve is optimizing the number of k-folds to 
    get the highest score possible in a kNN algorithm that we are running on the
    iris dataset.  In the above example, I simply try a bunch of different k-folds
    from 2-100 to see which yield the highest scores.

2) Problems that may arise --
    Because I force the range to be explicitly spelled out, we would not discover
    the correct answer if the correct number of folds were 101.  Additionally, this
    way of getting to the answer is a brute force method that is computationally
    very intensive.  Even for testing 2-100 k-folds, the computer clearly takes
    some time to produce an answer.
    My answer also seems to provide a different k-fold as the 'max' value each time
    I run it.  Sometimes it may say 95 folds is the right answer, sometimes 97.  
    Perhaps the real answer is to limit the number of folds to far less than 100 
    so we are mapping appropriately to the amount of data we have.
"""
=======
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

KNN=True 
NB=False
def load_iris_data() :

    # load the iris dataset from the sklearn module
    iris = datasets.load_iris()

    # extract the elements of the data that are used in this exercise
    return (iris.data, iris.target, iris.target_names)


def knn(X_train, y_train, k_neighbors = 3 ) :
    # function returns a kNN object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf


def nb(X_train, y_train) :
    # this function returns a Naive Bayes object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

# generic cross validation function
def cross_validate(XX, yy, classifier, k_fold) :

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :

        model = classifier(XX[[ train_slice  ]],
                         yy[[ train_slice  ]])

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold
>>>>>>> a9e85340bb201f4620ed62c91ce619caa006468e
