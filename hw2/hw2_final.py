from lesson1 import load_iris_data, cross_validate, knn, nb, lr, logregression

(XX,yy,y)=load_iris_data()

best_cv = 0

for c in range(1,10):
# Run logistic regression with C varying from 1-10    
    print "\n"
    print "--- %s ---" % c
    print "\n"
    
    # Run cross validation, holding constant at 5 folds

    cv = cross_validate(XX, yy, logregression, k_fold=5)
    print cv

    if cv > best_cv:
        best_cv = cv
        best_c = c
    
    print "The best C is %s with a CV score of %s" %(c, best_cv)
