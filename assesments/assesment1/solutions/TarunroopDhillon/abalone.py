#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:44:43 2021

@author: z5251296
"""

# MATH5836 Assessment 1
# z5251296

#Data Manipulation
import numpy as np
import pandas as pd

#Scikit learn
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Visualisation
import matplotlib.pyplot as plt
#import matplotlib as mpl
#%matplotlib inline
import seaborn as sns

#Statistics
from scipy.stats import skew

##### DATA PROCESSING #####

#Read in data
abalone_data_df = pd.read_table('data/abalone.data', delimiter=",", header=None)


## Part 1 ##
def data_clean(abalone_data_df, col_index):
    abalone_data_df.iloc[:,col_index] = abalone_data_df.iloc[:,col_index].map({'M':0, 'F':1, 'I':-1})
    return abalone_data_df

abalone_data = data_clean(abalone_data_df, 0)
# Convert to numpy array
abalone_data = np.array(abalone_data_df, dtype=float)

## Part 2 ##

def corr_map(abalone_data_array):

    corr_matrix = np.corrcoef(abalone_data_array.T)
    plt.draw()
    sns.heatmap(corr_matrix, annot=True, fmt='.2g')
    plt.title('Correlation Heatmap for Abalone Data')
    plt.savefig('feature_correlations.png')
    #plt.show()

corr_map(abalone_data)

'''
* There are no negative correlations in the heatmap.
* Features with index 1 to 7, i.e. Length, Diametre, Height, Whole weight, Shucked weight, Viscera weight and Shell   weight all can be observed to have high positive correlations amongst each other. This makes sense as all these     are measurements of the size of the Abalone. For example, if the height of Abalone 1 is more than the height of     Abalone 2, its whole weight is also likely to be greater than that of Abalone 2, so as height increases, whole       weight increases aswell and as such we observe the high correlation between them of 0.82. Similar can be said of     about other features mentioned above.
* None of the features are negatively correlated with the ring-age, and also none of the features have particularly   high positive correlations with ring-age.
'''


## Part 3 ##

'''
* There are no negative correlations so we can pick the features with the 2 highest positive correlations.
* These are diametre with a correlation of 0.57 with the target variable and shell weight with correlation of 0.63.
* Index are: Diametre: 2 and Shell weight: 7
'''

feature_dict = {0:'Sex', 1:'Length', 2:'Diametre', 3:'Height', 4:'Whole weight', 5:'Shucked weight', 6:'Viscera weight', 7:'Shell weight', 8:'Ring-age'}

def plot_scatter(data_array = abalone_data, feature_idx = [], target_idx = 8):
    for f in feature_idx:
        plt.draw()
        plt.scatter(data_array[:,f], data_array[:,target_idx], color='green')
        plt.xlabel(feature_dict[f])
        plt.ylabel(feature_dict[target_idx])
        plt.title(f'Scatterplot of {feature_dict[target_idx]} vs {feature_dict[f]}')
        plt.savefig('scatterplot_feature_' + f'{f}.png')
        #plt.show()

plot_scatter(data_array = abalone_data, feature_idx = [2,7], target_idx = 8)

'''
* Broadly speaking the feature values seem to be increasing with ring-age linearly.
* However, there seems to be an 'increasing funnel' effect which means that as diametre
  or shell weight increases, the ring-age variable increases but with increasing variability.
* As such, we could consider using more complex linear models such as generalised linear models,
  poisson regression is an option because a feature of the Poisson distributed response is increasing variance as     its mean increases.
* We can also see in the scatterplot of Ring-age vs Shell weight that there is a slight concave curvature. As such     we could try to capture this by including a quadratic polynomial term for shell weight in our model.
'''

## Part 4 ##
plt.style.use('ggplot')
def plot_hist(data_array = abalone_data, feature_idx = [], target_idx = 8, bins = 10):
    
    for f in feature_idx + [target_idx]:
        plt.draw()
        plt.hist(data_array[:,f], bins = bins, edgecolor = 'black', color = 'pink', alpha = 0.6)
        plt.xlabel(feature_dict[f])
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature_dict[f]}')
        plt.savefig(f'{feature_dict[f]}_hist.png')
        #plt.show()

plot_hist(data_array = abalone_data, feature_idx = [2,7], target_idx = 8, bins = 15)


print('Skew for diametre, shell weight and ring-age respectively is:')
print(skew(abalone_data[:,2]))
print(skew(abalone_data[:,7]))
print(skew(abalone_data[:,8]))

'''
* The histogram for diametre displays negative skewness.
* Can be confirmed with skew(abalone_data[:,2]) = -0.61
* The histogram for shell weight displays positive skewness.
* Can be confirmed with skew(abalone_data[:,7]) = 0.62
* The histogram for the target variable ring-age displays positive skewness.
* Can be confirmed with skew(abalone_data[:,8]) = 1.11
'''

## Part 5 ##
def split_data(abalone_data_array, train_proportion = 0.6, seed_i = 0, y_index = 8, normalise=True, features_to_use_index = 'all'):
    
    y = abalone_data_array[:, y_index]
    
    #Features to use
    if features_to_use_index == 'all':
        features = abalone_data_array[:, :y_index]
    else:
        features = abalone_data_array[:, features_to_use_index]
        
    #Normalise before splitting data
    if normalise:
        transformer = Normalizer().fit(features)
        features = transformer.transform(features)        
    
    x_train, x_test, y_train, y_test = train_test_split(features, y, train_size = train_proportion, random_state = seed_i, shuffle = True)
    return x_train, x_test, y_train, y_test



## Part 6 ##

#We can make use of Pair-wise Scatter Plots for the numerical vairables. 
# SOURCE: https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
def pairplot(abalone_data):
    
    plt.draw()
    pplot = sns.pairplot(abalone_data_df.iloc[:,1:], height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
    fig = pplot.fig 
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle('Abalone Attributes Pairwise Plots', fontsize=14)
    fig.savefig('Abalone Attributes Pairwise Plots.png')
    #fig.show()


pairplot(abalone_data)

'''
* The pair plot gives a very good visual summary of all the variables.
* We can notice that the variables are all positively aligned with each other.
* However, we can also notice curvature indicating that including higher order polynomial terms may be necessary to   increase the predictive power of the model.
* Along the diagonal are the kernel smoothed histograms which clearly exhibit skew in the variables.
* Neary all the variables display a 'funnel effect' with the response variable.
'''

#SOURCE:https://seaborn.pydata.org/generated/seaborn.boxplot.html
plt.draw()
ax = sns.boxplot(x=0, y=8, data=abalone_data_df.iloc[:,[0,8]]).set(xlabel='Sex', ylabel='Ring-Age', title = 'Boxplot for Sex of Abalone')

'''
* It can be clearly seen that ring-age is lower for the infant category as it should be.
* We can also notice the outliers which lie outside the whiskers.
'''


##### MODELLING #####

## Part 1 ##

def fit_lin_mod_and_resid(abalone_data, train_proportion = 0.6, seed_i = 0, y_index = 8, normalise=False, features_to_use_index = 'all'):
    
    x_train, x_test, y_train, y_test = split_data(abalone_data, train_proportion = train_proportion, seed_i = seed_i,\
                                                  y_index = y_index, normalise=normalise, features_to_use_index = features_to_use_index)
    
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    residuals = y_pred - y_test
    plt.draw()
    plt.plot(residuals, linewidth=1)
    plt.title('Residual Plot')
    plt.savefig('residual_plot.png')
    #plt.show()
    
    return rmse, r2

rmse, r2 = fit_lin_mod_and_resid(abalone_data, train_proportion = 0.6, seed_i = 0, y_index = 8, normalise=False, features_to_use_index = 'all')

'''
* The residuals show a random scatter around the horizontal axis and no discernable pattern which means that a linear model may be appropriate.
* The R-squared score is 0.5 which tells us that the model is not capturing a lot of the variability in the model.
'''

print(f'RMSE is {round(rmse,3)}')
print(f'R-sqaure is {round(r2,3)}')

## Part 2 ##

#Fitting linear regression model using all features (30 repititions).
def fit_lin_mod(abalone_data, train_proportion = 0.6, seed_i = 0, y_index = 8, normalise=False, features_to_use_index = 'all'):
    
    x_train, x_test, y_train, y_test = split_data(abalone_data, train_proportion = train_proportion, seed_i = seed_i,\
                                                  y_index = y_index, normalise=normalise, features_to_use_index = features_to_use_index)
    
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return rmse, r2


total_rep = 30
rmse_list = np.zeros(total_rep)
r2_list = np.zeros(total_rep)
rmse_n_list = np.zeros(total_rep)
r2_n_list = np.zeros(total_rep)

for rep in range(total_rep):
    
    #All input features without normalising
    rmse, r2 = fit_lin_mod(abalone_data, train_proportion = 0.6, seed_i = rep, y_index = 8,\
                           normalise=False, features_to_use_index = 'all')
    rmse_list[rep] = rmse
    r2_list[rep] = r2
    
    #All input features with normalising
    rmse_n, r2_n = fit_lin_mod(abalone_data, train_proportion = 0.6, seed_i = rep, y_index = 8,\
                               normalise=True, features_to_use_index = 'all')
    rmse_n_list[rep] = rmse_n
    r2_n_list[rep] = r2_n

print('Mean of RMSE using all input features and without normalising is: ' + f'{round(np.mean(rmse_list),3)}')
print('Standard deviation of RMSE using all input features and without normalising is: ' + f'{round(np.std(rmse_list),3)}')
print('Mean of R2-score using all input features and without normalising is: ' + f'{round(np.mean(r2_list),3)}')
print('Standard deviation of R2-score using all input features and without normalising is: ' + f'{round(np.std(r2_list),3)}')
print()
print()
print('Mean of RMSE using all input features and with normalising is: ' + f'{round(np.mean(rmse_n_list),3)}')
print('Standard deviation of RMSE using all input features and with normalising is: ' + f'{round(np.std(rmse_n_list),3)}')
print('Mean of R2-score using all input features and with normalising is: ' + f'{round(np.mean(r2_n_list),3)}')
print('Standard deviation of R2-score using all input features and with normalising is: ' + f'{round(np.std(r2_n_list),3)}')


## Part 3 and 4 ##

#Fitting linear regression model using only diametre and shell weight (30 repititions).
total_rep = 30
rmse_list = np.zeros(total_rep)
r2_list = np.zeros(total_rep)
rmse_n_list = np.zeros(total_rep)
r2_n_list = np.zeros(total_rep)

for rep in range(total_rep):
    
    #2 input features without normalising
    rmse, r2 = fit_lin_mod(abalone_data, train_proportion = 0.6, seed_i = rep, y_index = 8, normalise=False, features_to_use_index = [2,7])
    rmse_list[rep] = rmse
    r2_list[rep] = r2
    
    #2 input features with normalising
    rmse_n, r2_n = fit_lin_mod(abalone_data, train_proportion = 0.6, seed_i = rep, y_index = 8, normalise=True, features_to_use_index = [2,7])
    rmse_n_list[rep] = rmse_n
    r2_n_list[rep] = r2_n


print('Mean of RMSE using Diametre and Shell weight as input features and without normalising is: ' + f'{round(np.mean(rmse_list),3)}')
print('Standard deviation of RMSE using Diametre and Shell weight as input features and without normalising is: ' + f'{round(np.std(rmse_list),3)}')
print('Mean of R2-score using Diametre and Shell weight as input features and without normalising is: ' + f'{round(np.mean(r2_list),3)}')
print('Standard deviation of R2-score using Diametre and Shell weight as input features and without normalising is: ' + f'{round(np.std(r2_list),3)}')
print()
print()
print('Mean of RMSE using Diametre and Shell weight as input features and with normalising is: ' + f'{round(np.mean(rmse_n_list),3)}')
print('Standard deviation of RMSE using Diametre and Shell weight as input features and with normalising is: ' + f'{round(np.std(rmse_n_list),3)}')
print('Mean of R2-score using Diametre and Shell weight as input features and with normalising is: ' + f'{round(np.mean(r2_n_list),3)}')
print('Standard deviation of R2-score using Diametre and Shell weight as input features and with normalising is: ' + f'{round(np.std(r2_n_list),3)}')


'''
Discussion

* It can be seen that when we used all the variables in the model and when we used only diametre and shell weight in the model; for both of these models, the mean rmse was higher when we did not normalise the data as compared to when we did. Also, the mean r squared score was higher when we normalised the data as compared to when we did not.

* We can see that the model with all the variables used had a lower mean rmse than the model with only the 2 selected features in the model; this is the case for both with and without normalising the data.

* We can see that the model with all the variables used had a higher mean r squared score than the model with only the 2 selected features in the model; this is the case for both with and without normalising the data.

* This means that when we use only diametre and shell weight variables in the model, even though these were selected based on them having high positive correlations with the response variable, the model performs worse becuase it is not capturing as much variability inherent in the data as the model with all variables is. A solution to this is to include more variables or select them using techniques such as forward/backward feature selection.

* We can see that the standard deviation of the rmse scores is higher for the model using only diametre and shell weight as compared to the model with all variables; this is the case for both with and without normalising the data. One reason for this could be that these two variables are highly correlated with each other (0.91) which introduces multi-collinearity problem. This causes more variability in the predictions made by the model reflected in higher rmse and higher variability in rmse as the results above show. However, as noted in the comments under the correlation matrix, all variables (except sex) exhibit high correlations with each other.
'''












