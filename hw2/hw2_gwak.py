from hw2 import load_iris_data, cross_validate, knn, nb, lr

(XX,yy,y)=load_iris_data()

# For HW2, I am only running the linear regression

best_k=0
best_cv=0

foldset = [2,3,5,10,20,30,40,50,75]

for i in foldset:
    cv_a = cross_validate(XX,yy,lr,k_fold=i)
    if cv_a > best_cv:
        best_cv=cv_a
        best_k=i

        print "fold <<%s>> :: acc <<%s>>" % (i, cv_a)
