import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import random


from sklearn.metrics import roc_curve, auc

def read_data(run_num):
    #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

    data_in = genfromtxt("pima.csv", delimiter=",")
 
    data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7 

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    #transformer = Normalizer().fit(data_inputx)  # fit does nothing.
    #data_inputx = transformer.transform(data_inputx)

    data_inputy = data_in[:,-1] # this is target - so that last col is selected from data
 
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)

    return x_train, x_test, y_train, y_test

 
    
def scipy_nn(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num):
    #Source: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

    #random_stateint, RandomState instance, default=None Determines random number generation
    # for weights and bias initialization, train-test split if early stopping is used, and batch sampling when solver=’sgd’
    #or ‘adam’. Pass an int for reproducible results across multiple function calls.

    #learning_rate_initdouble, default=0.001
    #The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.

    #note Adam does not need momentum and constant learning rate since they are adjusted in Adam itself








    if type_model ==0: #SGD
        nn = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100,solver='sgd',  learning_rate_init=learn_rate )
        #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    elif type_model ==1: #Adam
        nn = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100,solver='adam', learning_rate_init=learn_rate)
        #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    elif type_model ==2: #SGD with 2 hidden layers
        nn = MLPClassifier(hidden_layer_sizes=(hidden,hidden), random_state=run_num, max_iter=100,solver='sgd',learning_rate='constant', learning_rate_init=learn_rate)
        #hidden_layer_sizes=(hidden,hidden, hidden) would implement 3 hidden layers
    else:
        print('no model')    
 
    # Train the model using the training sets
    nn.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred_test = nn.predict(x_test)
    y_pred_train = nn.predict(x_train)

    #print([coef.shape for coef in nn.coefs_], 'weights shape')
 
    #print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))  
    acc_test = accuracy_score(y_pred_test, y_test) 
    acc_train = accuracy_score(y_pred_train, y_train) 

    cm = confusion_matrix(y_pred_test, y_test) 
    #print(cm, 'is confusion matrix')

    #auc = roc_auc_score(y_pred, y_test, average=None) 
    return acc_test #,acc_train


def main(): 

    max_expruns = 5

    SGD_all = np.zeros(max_expruns) 
    Adam_all = np.zeros(max_expruns) 
    
    SGD2_all = np.zeros(max_expruns)  
    max_hidden = 12

    learn_rate = 0.01
    #hidden = 8


    #for learn_rate in range(0.1,1, 0.2):
    
    for hidden in range(6,max_hidden, 2):
 
        for run_num in range(0,max_expruns): 
    
            x_train, x_test, y_train, y_test = read_data(0)   
            
            acc_sgd = scipy_nn(x_train, x_test, y_train, y_test, 0, hidden, learn_rate, run_num) #SGD
            acc_adam = scipy_nn(x_train, x_test, y_train, y_test, 1, hidden, learn_rate, run_num) #Adam 
            acc_sgd2 = scipy_nn(x_train, x_test, y_train, y_test, 2, hidden, learn_rate,  run_num) #SGD2
           
            SGD_all[run_num] = acc_sgd
            Adam_all[run_num] = acc_adam

            SGD2_all[run_num] = acc_sgd2 # two hidden layers
        
        print(SGD_all, hidden,' SGD_all')
        print(np.mean(SGD_all), hidden, ' mean SGD_all')
        print(np.std(SGD_all), hidden, ' std SGD_all')

        print(Adam_all, hidden,' Adam_all')
        print(np.mean(Adam_all), hidden, ' Adam _all')
        print(np.std(Adam_all), hidden, ' Adam _all')

        print(SGD2_all, hidden,' SGD2_all')

        # you can also print  for Adam

 
 
    #next try a paragraph to describe your results and discuss which models are better to use.
    #repeat for another dataset
    # you can save results to a file as well


 


if __name__ == '__main__':
     main() 
