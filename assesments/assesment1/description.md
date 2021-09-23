## Option I: Data processing and linear regression
 Refer to the dataset below.

### Abalone Dataset: Predict the Ring age in years

"Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem. From the original data examples with missing values were removed (the majority having the predicted value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by 200)." 

Source: https://archive.ics.uci.edu/ml/datasets/abalone

Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter / continuous / mm / perpendicular to length
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years (ring-age)

Ignore the +1.5 in ring-age and use the raw data

### Data processing (1.5 Marks):

* Clean the data (eg. convert M and F to 0 and 1). You can do this with code or simple find and replace. 
* Develop a correlation map using a heatmap and discuss major observations.
* Pick two of the most correlated features (negative and positive) and create a scatter plot with ring-age. Discuss major observations. 
* Create histograms of the two most correlated features along with the ring-age. What are the major observations? 
* Create a 60/40 train/test split - which takes a random seed based on the experiment number to create a new dataset for every experiment. 
* Add any other visualisation of the dataset you find appropriate. 

### Modelling  (2.0 Marks):

* Develop a linear regression model using all features for ring-age using 60 percent of data picked randomly for training and remaining for testing. Visualise your model prediction using appropriate plots. Report the RMSE and R-squared score. 
* Develop a linear regression model with all input features, i) without normalising input data, ii) with normalising input data. 
* Develop a linear regression model with two selected input features from the data processing step. 
* In each of the above investigations, run 30 experiments each and report the mean and std of the RMSE and R-squared score of the train and test datasets. Write a paragraph to compare your results of the different approaches taken.
* Upload your code in Python/R or both. The code should use relevant functions (or methods in case you use OOP) and then create required outputs. 

### Report  (1.5 Marks):

* Create a report and include the visualisations and results obtained and discuss the major trends you see in the visualisation and modelling by linear regression  
* Upload a pdf of the report. 


## Option II: Data essay
