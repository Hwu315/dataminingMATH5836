import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import random


from sklearn.metrics import roc_curve, auc

def read_data(run_num):
    #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

    #data_in = genfromtxt("irisprocessed.csv", delimiter=",")

    data_in = genfromtxt("pima-indians-diabetes.csv", delimiter=",")
 
    data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7
    #data_inputx = data_in[:,[1]]  # one feature
    #data_inputx = data_in[:,[1,2]]  # two features

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    #transformer = Normalizer().fit(data_inputx)  # fit does nothing.
    #data_inputx = transformer.transform(data_inputx)

    data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

    #percent_test = 0.4
    #testsize = int(percent_test * data_inputx.shape[0]) 
    #x_train = data_inputx[:-testsize]
    #x_test = data_inputx[-testsize:] 
    #y_train = data_inputy[:-testsize]
    #y_test = data_inputy[-testsize:]

    #another way you can use scikit-learn train test split with random state
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)


    return x_train, x_test, y_train, y_test

 
    
def scipy_linear_mod(x_train, x_test, y_train, y_test, type_model):
    #Source: Scikit Learn. (n.d). Linear Regression Example. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 
  
    if type_model ==0:
        regr = linear_model.LogisticRegression(tol=0.01,solver='saga') #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    elif type_model ==1: #L1 regu
        regr = linear_model.LogisticRegression(penalty='l1', tol=0.01,solver='saga')
    else:
        regr = linear_model.LogisticRegression(penalty='l2', tol=0.01,solver='saga') # L2 regu
        
        #saga: Stochastic Average Gradient descent 
        #tol:  Tolerance for stopping criteria.

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
 
 
    acc = accuracy_score(y_pred, y_test)

    print(acc, ' is accuracy_score') 
    cm = confusion_matrix(y_pred, y_test)

    print(cm, 'is confusion matrix')

    auc = roc_auc_score(y_pred, y_test, average=None)

    print(auc, " is AUC ")

    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    # plot the roc curve for the model 
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic-model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(str(type_model)+'plot.png')
    plt.clf()

    #cross validation can also be done on test and train dataset combined.
    #Here I am using only the train dataset, hence part of the train dataset is used for test or validation
    cv_results = cross_validate(regr, x_train, y_train, cv=10)  
    print(cv_results, ' of 10 fold cross validation')
    #refer to https://edstem.org/courses/5850/lessons/10369/slides/75708

    return auc, acc


def main(): 

    max_expruns = 3

    accbase_all = np.zeros(max_expruns) 
    accl1_all = np.zeros(max_expruns) 
    accl2_all = np.zeros(max_expruns)


 

    for run_num in range(0,max_expruns):
 
        x_train, x_test, y_train, y_test = read_data(run_num)  
        #print(x_train, ' x_train')
        #print(y_train, ' y_train')  
        
        auc_base, acc_base = scipy_linear_mod(x_train, x_test, y_train, y_test, 0) #base model
        auc_l1, acc_l1 = scipy_linear_mod(x_train, x_test, y_train, y_test, 1) #L1 regu
        auc_l2, acc_l2 = scipy_linear_mod(x_train, x_test, y_train, y_test, 2) #L2 regu

        accbase_all[run_num] = acc_base
        accl1_all[run_num] = acc_l1
        accl2_all[run_num] = acc_l2


        print(run_num, ' -- run num --')
        print(acc_base, acc_l1, acc_l2, ' acc_base, acc_l1, acc_l2')
    
    print(accbase_all, ' accbase_all')
    print(np.mean(accbase_all), ' mean accbase_all')
    print(np.std(accbase_all), ' std accbase_all')

    # you can also print  for l1 and l2


    print(accl1_all, ' accl1_all')
    print(np.mean(accl1_all), ' mean accl1_all')
    print(np.std(accl1_all), ' std accl1_all')
 
    #next try a paragraph to describe your results and discuss which models are better to use.
    #repeat for another dataset
    # you can save results to a file as well


 


if __name__ == '__main__':
     main() 
