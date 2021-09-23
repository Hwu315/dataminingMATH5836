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


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
 
from sklearn import metrics
from sklearn.metrics import roc_auc_score


def get_data(data, normalise, i): 

    if data == 1:
        data_in = genfromtxt('data/iris_binaryenc.csv', delimiter=",") #  iris - binary problem
        data_inputx = data_in[:,0:4]  # two features   
        data_inputy = data_in[:,4] # this is target - binary encoding (2 classes)

    elif data == 2:
        data_in = genfromtxt('data/data_banknote_authentication.txt', delimiter=",") # bank note prob 
        data_inputx = data_in[:,0:4]  # two features   
        data_inputy = data_in[:,4] # this is target - one hot encoing  (3 classes)
        #https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    
    elif data == 3:
        data_in = genfromtxt('data/processed.cleveland.data', delimiter=",") #  make this work for cleveland heart problem  
        data_inputx = data_in[:,0:13]  #  few rows had ? which i replaced with 0
        data_inputy = data_in[:,13] # set for wine  
        data_inputy = np.digitize(data_inputy,bins=[1])  # 1 and above are 1, 0 remains 0

        #https://archive.ics.uci.edu/ml/datasets/heart+disease

        #next try : https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
        #https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

    else: 
        print(' select data')

    print(data_inputx)
 


    if normalise == True:
        transformer = Normalizer().fit(data_inputx)   
        data_inputx = transformer.transform(data_inputx)
 


    percent_test = 0.4 
 
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=percent_test, random_state=i)

    return x_train, x_test, y_train, y_test
     
def scikit_linear_mod(x_train, x_test, y_train, y_test): 
 
    regr = linear_model.LogisticRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)



    # Make predictions using the testing set
    y_pred_test = regr.predict(x_test)
    y_pred_train = regr.predict(x_train)
 
  
    accuracy_test = accuracy_score(y_test, y_pred_test) 
    accuracy_train = accuracy_score(y_train, y_pred_train)

    cm = confusion_matrix(y_pred_test, y_test)  

    print(cm, ' is cm')

    return accuracy_train, accuracy_test, regr.coef_




def main(): 

    normalise = False
 
    
    max_exp = 10

    rmse_list = np.zeros(max_exp)
    rsq_list = np.zeros(max_exp)

    data = 3 # for iris binary, 2 for iris three class, 3 for wine 

    for i in range(0,max_exp):
        
        x_train, x_test, y_train, y_test = get_data(data, normalise, i)
        rmse, rsquared, coef = scikit_linear_mod(x_train, x_test, y_train, y_test)
        
        rmse_list[i] = rmse
        rsq_list[i] = rsquared 
        

    print(rmse_list)
    # Explained variance score: 1 is perfect prediction 
    print(rsq_list)
    
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    mean_rsq = np.mean(rsq_list)
    std_rsq = np.std(rsq_list)

    print(mean_rmse, std_rmse, ' mean_rmse std_rmse')
    

    print(mean_rsq, std_rsq, ' mean_rsq std_rsq')



  

if __name__ == '__main__':
     main()

