import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


import random

 

 

def read_data():
    #Source: University of California. (n.d). Machine-learning-databases. http://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    #Source: University of California. (n.d). Machine learning repository. http://archive.ics.uci.edu/ml/datasets/iris 
    #Source: Iris flower dataset. (2020). Wikipedia. https://en.wikipedia.org/wiki/Iris_flower_data_set

 

    data_in = genfromtxt("data/iris_binaryenc.csv", delimiter=",")
 
    data_inputx = data_in[:,0:4] # all features 0, 1, 2, 3

    #data_inputx = data_in[:,[1]]  # one feature

    #data_inputx = data_in[:,[1,2]]  # two features

    data_inputy = data_in[:,-1] # this is target - so that last col is selected from data
 
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=0)


    return x_train, x_test, y_train, y_test
 
    
def linear_mod(x_train, x_test, y_train, y_test):
    #Source: Scikit Learn. (n.d). Linear Regression Example. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 
 
    print(' running scipy linear model')

    regr = linear_model.LinearRegression()


    # Create linear regression object

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction (correlation)
    #mse = mean_squared_error()
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    print(y_pred, ' y_pred')

    #y_pred_int = y_pred.astype(int) # n case you want to just convert to int

    #y_pred_int = np.rint(y_pred) # round off and convert to int

    #https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
    y_pred_int = np.heaviside(y_pred, 1)

    print(y_pred_int, ' y_pred int ')

    print(y_test, ' y_test')

    acc = accuracy_score(y_pred_int, y_test)

    print(acc, ' is accuracy_score')
 

    return acc 




def main(): 
 
 
    x_train, x_test, y_train, y_test = read_data() # when you read from file 

    print(x_train, ' x_train')
    print(y_train, ' y_train') 


    acc = linear_mod(x_train, x_test, y_train, y_test)
 


if __name__ == '__main__':
     main()

