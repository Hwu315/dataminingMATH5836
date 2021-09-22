
import numpy as np 

import matplotlib.pyplot as plt


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


from sklearn import linear_model

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
 

#http://archive.ics.uci.edu/ml/machine-learning-databases/housing/

#http://archive.ics.uci.edu/ml/datasets/iris
#https://en.wikipedia.org/wiki/Iris_flower_data_set  



def get_data_diabetes():
  
    normalise = True

    diabetes = datasets.load_diabetes() 

    data_inputx = diabetes.data 
 
    if normalise == True:
        transformer = Normalizer().fit(data_inputx)   
        data_inputx = transformer.transform(data_inputx)
  
 
    data_y = diabetes.target

    percent_test = 0.4 

      #another way you can use scikit-learn train test split with random state
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_y, test_size=percent_test, random_state=0)

    return x_train, x_test, y_train, y_test



def get_data_iris():
  
    iris = datasets.load_iris() # https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

    data_input = iris.data[:, np.newaxis, 2] 

    x_train = data_input[:-60]
    x_test = data_input[-60:]

    # Split the targets into training/testing sets
    y_train = iris.target[:-60]
    y_test = iris.target[-60:]

    # Split the data into training/testing sets

    return x_train, x_test, y_train, y_test

def linear_scikit(x_train, x_test, y_train, y_test):

    regr = linear_model.LinearRegression()
    regr.fit(x_train,  y_train) 
    print(regr.coef_) 
    error = np.mean((regr.predict(x_test) - y_test)**2) 
    print(error, 'is error') 
    r_score = regr.score(x_test, y_test)  # Explained variance score: 1 is perfect prediction
                                                  # and 0 means that there is no linear relationshipbetween X and y.

    print(r_score, 'is R score')



def logistic_scikit(x_train, x_test, y_train, y_test):

     
    logistic = linear_model.LogisticRegression()
    logistic.fit(x_train, y_train)  

    print(logistic.coef_) 

    y_pred_train = logistic.predict(x_train)
    y_pred_test = logistic.predict(x_test)

    error = np.mean((y_pred_test - y_test)**2) 
    print(error, 'is error') 
    r_score = logistic.score(x_test, y_test)  

    print(r_score, 'is R score')  # not appropiate measure of error for classification probs
    acc_test = accuracy_score(y_pred_test, y_test) 
    acc_train = accuracy_score(y_pred_train, y_train) 
    cm = confusion_matrix(y_pred_test, y_test) 

    print(acc_test, acc_train, ' * a test and train')
    print(cm, ' confusion mat')

 

 


def main(): 

    
    #x_train, x_test, y_train, y_test = get_data_iris()

    x_train, x_test, y_train, y_test = get_data_diabetes()

    print(x_test.shape, ' x_train')


    print( ' run linear regression')

    linear_scikit(x_train, x_test, y_train, y_test)


    print( ' run logistic regression')

 
    logistic_scikit(x_train, x_test, y_train, y_test)

     
 

if __name__ == '__main__':
    main()

