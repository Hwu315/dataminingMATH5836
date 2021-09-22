
from sklearn import datasets, linear_model

from sklearn.model_selection import cross_validate

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target
# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the accuracy changes a lot
# this is why testing accuracy is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)



#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
# simulate splitting a dataset  into 5 folds
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]

y = diabetes.target[:150]
lasso = linear_model.Lasso()
#Single metric evaluation using cross_validate

cv_results = cross_validate(lasso, X, y, cv=5)
res = sorted(cv_results.keys()) 
print(res)
ind_res = cv_results['test_score'] 
print(ind_res, ' CV res')

scores = cross_validate(lasso, X, y, cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
print(scores['test_neg_mean_squared_error']) 
print(scores['train_r2'])
