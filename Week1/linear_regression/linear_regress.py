import numpy as np 

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random


from numpy import *  
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split


from scipy.stats import pearsonr

def get_data(normalise):
    #Source: University of California. (n.d). Machine-learning-databases. http://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    #Source: University of California. (n.d). Machine learning repository. http://archive.ics.uci.edu/ml/datasets/iris 
    #Source: Iris flower dataset. (2020). Wikipedia. https://en.wikipedia.org/wiki/Iris_flower_data_set
 
    #house_data = datasets.load_boston() #Scikit-learn provides a handy description of the dataset, and it can be easily viewed by:


    data_in = genfromtxt('raw_data/housing.data')

    #data_in = genfromtxt('raw_data/housing.data', delimiter=",") # in case of csv data

    #print(data_in)

    #data_inputx = data_in[:,0:13] # all features 0, - 12
    #data_inputx = data_in[:,[1]]  # one feature
    data_inputx = data_in[:,[5,12]]  # two features (RM LSTAT)

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer

    if normalise == True:
        transformer = Normalizer().fit(data_inputx)  # fit does nothing.
        data_inputx = transformer.transform(data_inputx)
 

    #cov_mat = np.cov(data_in.T)

    corr_mat = np.corrcoef(data_in.T)

    #print(corr_mat, ' is the corr matrix of the data read')

    plt.imshow(corr_mat, cmap='hot', interpolation='nearest')
    plt.savefig('cov_heatmap.png')
    plt.clf()

    data_inputy = data_in[:,13] # this is target - so that last col is selected from data

    percent_test = 0.4
    '''testsize = int(percent_test * data_inputx.shape[0]) 
    x_train = data_inputx[:-testsize]
    x_test = data_inputx[-testsize:] 
    y_train = data_inputy[:-testsize]
    y_test = data_inputy[-testsize:]'''


      #another way you can use scikit-learn train test split with random state
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=percent_test, random_state=0)



    return x_train, x_test, y_train, y_test
  

    
def scikit_linear_mod(x_train, x_test, y_train, y_test):
    #source: Scikit Learn. (n.d). Linear Regression Example. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 

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
    print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f' % r2_score(y_test, y_pred))

    # Plot residuals

    residuals = y_pred - y_test
    plt.plot(residuals, linewidth=1)
 
    plt.savefig('scikit_linear.png')




def main(): 

    normalise = False

    x_train, x_test, y_train, y_test = get_data(normalise)

    #print(x_train, ' x_train')
    #print(y_train, ' y_train')
    #print(x_test, ' x_test')


    scikit_linear_mod(x_train, x_test, y_train, y_test)
 
    numpy_linear_mod(x_train, x_test, y_train, y_test)











if __name__ == '__main__':
     main()
