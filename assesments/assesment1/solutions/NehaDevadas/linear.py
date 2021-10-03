import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
from numpy import *  
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import pearsonr
# from yellowbrick.regressor import PredictionError, ResidualsPlot

def import_data():
    df = pd.read_csv('data/abalone.data',delimiter=',',header=None)
    data_in = df.values
#     print("Data before processing is \n")
#     print(data_in)
    data_in[:,0] = np.where(data_in[:,0] == 'M', 0, np.where(data_in[:,0] == 'F', 1, -1))
#     print("Data after processing is \n")
#     print(data_in)
#     print("First column is now\n")
#     print(data_in[:,0])
#     print(data_in.shape)

    data_in = data_in.astype(float32)
    corr_mat = np.corrcoef(data_in.T)
    print(corr_mat, ' is the correlation matrix of the data')
    plt.figure(figsize=(5, 5))
    sns.heatmap(data=corr_mat, annot=True)
    plt.title("Correlation Matrix")
    plt.savefig('correlation_matrix_data.png')
    
    return data_in
    

def plot_feature(data_inputx,data_inputy):
    selected_features = data_inputx[:,[2,7]]
    for i in range(selected_features.shape[1]):
        x = selected_features[:,i]
#         print("x is \n",x.shape)
        y = data_inputy
#         print("y is \n",y.shape)
        plt.figure()
        plt.scatter(x, y, marker='o',edgecolor='black')
        plt.title('Feature '+str(i+1)+' vs Target')
        plt.xlabel('Feature '+str(i+1))
        plt.ylabel('Target')
        plt.savefig('Feature'+str(i+1)+'vsTarget'+'.png')
        
def hist_plot(name, data):
    plt.figure()
    plt.hist(data,color='purple',edgecolor='black')
    plt.title(name)
    plt.savefig(name+'.png')
    
def split_data(data_in, normalise, i,number_of_features): 

    
    if number_of_features == 2:
        print("Experiment with 2 features running")
        data_inputx = data_in[:,[2,7]]  # two features 2 and 7
    else:
        print("Experiment with all features running")
        data_inputx = data_in[:,0:8] # all features 0, - 7
    
    data_inputy = data_in[:,8] # target data
    
    if normalise == True: #------------------------------Normalisation
        transformer = Normalizer().fit(data_inputx)  
        data_inputx = transformer.transform(data_inputx)

    percent_test = 0.4

    #using scikit-learn train test split with random state
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=percent_test, random_state=i)

    return x_train, x_test, y_train, y_test
     
def scikit_linear_mod(x_train, x_test, y_train, y_test): 
 
    regr = linear_model.LinearRegression()

 
    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
    rsquared = r2_score(y_test, y_pred) 
#     plt.figure()
#     visualizer = PredictionError(regr)
#     visualizer.fit(x_train, y_train)  
#     visualizer.score(x_test, y_test)  
#     visualizer.poof()
#     plt.figure()
#     visualizer = ResidualsPlot(regr)
#     visualizer.fit(x_train, y_train)  
#     visualizer.score(x_test, y_test)  
#     visualizer.poof()
 
#     residuals = y_pred - y_test
#     plt.figure()
#     plt.plot(residuals, linewidth=1,color = 'green')
#     plt.title("Residual plot for 2 features with normalisation")
#     plt.savefig('Residual_plot_with_norm_2.png')


    return rmse, rsquared, regr.coef_




def main(): 
    
    data = import_data()
    data_inputx = data[:,:8]
    print("Features are \n")
    print(data_inputx)
    data_inputy = data[:,8]
    print("Target is \n")
    print(data_inputy)
    
    plot_feature(data_inputx,data_inputy)
    
    hist_plot("Shell weight",data_inputx[:,7])
    hist_plot("Diameter",data_inputx[:,2])
    hist_plot("Ring-age",data_inputy)
    
    #-------------------------------------------
    hist_plot("Length",data_inputx[:,1])
    hist_plot("Height",data_inputx[:,3])
    hist_plot("Whole weight",data_inputx[:,4])
    hist_plot("Shucked weight",data_inputx[:,5])
    hist_plot("Viscera weight",data_inputx[:,6])
    hist_plot("Sex",data_inputx[:,0])
    #-------------------------------------------
    normalise = True
    max_exp = 1 #can change it to 30
    number_of_features = 2 #Enter 2 for 2 features and 0 for all features
    rmse_list = np.zeros(max_exp) 
    rsq_list = np.zeros(max_exp)   

    for i in range(0,max_exp):
        
        x_train, x_test, y_train, y_test = split_data(data,normalise,i,number_of_features)
        rmse, rsquared, coef = scikit_linear_mod(x_train, x_test, y_train, y_test)
        
        rmse_list[i] = rmse
        rsq_list[i] = rsquared 
        

    print(rmse_list, 'is the RMSE list')
    print(rsq_list, 'is the R2 list')
    
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)

    mean_rsq = np.mean(rsq_list)
    std_rsq = np.std(rsq_list)

    print(mean_rmse, std_rmse, ' mean_rmse std_rmse')

    print(mean_rsq, std_rsq, ' mean_rsq std_rsq')
    


if __name__ == '__main__':
     main()

