
#Exercise 3: 
#https://github.com/vasilikivamv/MATH5836/tree/master/Week%201
#Source: Vasiliki Vamvaka Solution: https://github.com/vasilikivamv/MATH5836/blob/master/Week%201/exercise_3_iris_data.R
## Loading the data set

library(datasets)
data(iris)
attach(iris)

## Structure of the data set iris
str(iris)

summary(iris)

# Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
# Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
# 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
# Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
# Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
# 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
# Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  

#1.  transform the variable "Species" into a numeric value
iris$Species <- ifelse(Species == "setosa", 1, NA)
iris$Species <- ifelse(Species == "versicolor", 2, iris$Species)
iris$Species <- ifelse(Species == "virginica", 3, iris$Species)

#2. mean for each column
mean(Sepal.Length)  # 5.843333
mean(Sepal.Width)   # 3.057333
mean(Petal.Length)  # 3.758
mean(Petal.Width)   # 1.199333

# standard deviation for each column
sd(Sepal.Length)  #  0.8280661
sd(Sepal.Width)   #  0.4358663
sd(Petal.Length)  #  1.765298
sd(Petal.Width)   #  0.7622377


#3. frequency table of the classes
table(Species)

# Species (setosa=1, versicolor=2, virginica=3)
# setosa versicolor  virginica 
# 50         50         50 



#4. histogram of each feature
png(file = "histograms.png")
par(mfrow=c(2,2))
hist(Sepal.Length,main = "Histogram of Sepal Lenght",col = "blue")
hist(Sepal.Width,main = "Histogram of Sepal Width",col = "blue")
hist(Petal.Length,main = "Histogram of Petal Lenght",col = "blue")
hist(Petal.Width,main = "Histogram of Petal Width",col = "blue")

# Save the file.
dev.off()

#5. Spliting the data set to training and testing sets

# 60% of the data are assigned to the test data (randomly assigned) (i)
set.seed(123)
training.samples <- sample(1:nrow(iris), nrow(iris)*0.6, replace = F)

train.data  <- iris[training.samples, ]
test.data <- iris[-training.samples, ]

# first 60% of data are assigned as train data (ii)

nrow(iris)*0.6 #90 first rows are the 60% of the data
train.first60 <- iris[1:90,]
test.rest <-iris[91:150,]

#6. mean and sd of the test data set

#initialization of matrixes
mean_train <- matrix(NA, nrow = 1, ncol=4)
mean_test <- matrix(NA, nrow = 1, ncol=4 )
sd_train <- matrix(NA, nrow = 1, ncol=4)
sd_test <- matrix(NA, nrow = 1, ncol=4)

for (i in 1:4) {
  
  mean_train[,i] <- mean(train.data[,i])
  mean_test[,i] <- mean(test.data[,i])
  sd_train[,i] <- sd(train.data[,i])
  sd_test[,i] <- sd(test.data[,i])
  
}


#frequency table of the classes in the train data set
table(train.data$Species)

# Species (setosa=1, versicolor=2, virginica=3)
# 1     2      3 
# 30    26     34

#frequency table of the classes in the test data set
table(test.data$Species)

# Species (setosa=1, versicolor=2, virginica=3)
# 1     2      3 
# 20    24     16


#histograms of each feature
par(mfrow=c(4,2))

hist(train.data$Sepal.Length,main = "Histogram of Sepal Lenght,train",col = "blue")
hist(train.data$Sepal.Width,main = "Histogram of Sepal Width,train",col = "blue")
hist(train.data$Petal.Length,main = "Histogram of Petal Lenght,train",col = "blue")
hist(train.data$Petal.Width,main = "Histogram of Petal Width,train",col = "blue")


hist(test.data$Sepal.Length,main = "Histogram of Sepal Lenght,test",col = "green")
hist(test.data$Sepal.Width,main = "Histogram of Sepal Width,test",col = "green")
hist(test.data$Petal.Length,main = "Histogram of Petal Lenght,test",col = "green")
hist(test.data$Petal.Width,main = "Histogram of Petal Width,test",col = "green")

#7. part of train set with only one feature
one.featuretrain <- subset(train.data, train.data$Species == 1)

# part of test data with one feature
one.featuretest <- subset(test.data, test.data$Species == 1)


#8. binary problem

# remove one class
binary.iris <- iris[!(iris$Species == 3),] 


#Spliting the data set to training and testing sets

set.seed(123)
training.samples <- sample(1:nrow(binary.iris), nrow(binary.iris)*0.6, replace = F)

binary_iristrain  <- binary.iris[training.samples, ]
binary_iristest <- binary.iris[-training.samples, ]

#mean and sd of the test data set

#initialization of matrixes
mean_binary_iristrain <- matrix(NA, nrow = 1, ncol=4)
mean_binary_iristest <- matrix(NA, nrow = 1, ncol=4 )
sd_binary_iristrain <- matrix(NA, nrow = 1, ncol=4)
sd_binary_iristest <- matrix(NA, nrow = 1, ncol=4)

for (i in 1:4) {
  
  mean_binary_iristrain[,i] <- mean(binary_iristrain[,i])
  mean_binary_iristest[,i] <- mean(binary_iristest[,i])
  sd_binary_iristrain[,i] <- sd(binary_iristrain[,i])
  sd_binary_iristest[,i] <- sd(binary_iristest[,i])
  
}


#frequency table of the classes in the train data set
table(binary_iristrain$Species)

# Species (setosa=1, versicolor=2)
# 1      2   
# 30    30

#frequency table of the classes in the test data set
table(binary_iristest$Species)

# Species (setosa=1, versicolor=2)
# 1     2      
# 20    20    


#histograms of each feature
par(mfrow=c(4,2))

hist(binary_iristrain$Sepal.Length,main = "Histogram of Sepal Lenght,train",col = "blue")
hist(binary_iristrain$Sepal.Width,main = "Histogram of Sepal Width,train",col = "blue")
hist(binary_iristrain$Petal.Length,main = "Histogram of Petal Lenght,train",col = "blue")
hist(binary_iristrain$Petal.Width,main = "Histogram of Petal Width,train",col = "blue")


hist(binary_iristest$Sepal.Length,main = "Histogram of Sepal Lenght,test",col = "green")
hist(binary_iristest$Sepal.Width,main = "Histogram of Sepal Width,test",col = "green")
hist(binary_iristest$Petal.Length,main = "Histogram of Petal Lenght,test",col = "green")
hist(binary_iristest$Petal.Width,main = "Histogram of Petal Width,test",col = "green")


#9. normalized inputs between [0,1]

#normalization function
norm.function <- function(feature) {
  (feature - min(feature)) / (max(feature) - min(feature))
}

iris.normal <- as.data.frame (lapply(iris[1:4], norm.function ))

# add the Species feature again
iris.normal$Species <- iris$Species
head(iris.normal)