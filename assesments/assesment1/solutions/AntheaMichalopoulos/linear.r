library(ggplot2)
library(reshape2)
abalone <- read.table("data/abalone.data", sep = ",", header = F)
col.names <- c("Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings")

abalone_MF <- abalone

for (i in 1:nrow(abalone_MF)){
  if(abalone_MF[i,1] == "M"){
    abalone_MF[i,1] = 0
  }
  else if (abalone_MF[i,1] == "F"){
    abalone_MF[i,1] = 1
  }
  else{
    abalone_MF[i,1] = -1
  }
}
abalone_MF <- matrix(as.numeric(as.matrix(abalone_MF)), ncol = ncol(abalone_MF))
abalone_MF <- as.data.frame(abalone_MF)
colnames(abalone_MF) <- col.names 

cor.matrix <- round(cor(abalone_MF), digits = 2) 
cor.matrix
cor.matrix.melt <- melt(cor.matrix)

pdf("Correlation Heat Map.pdf")
ggplot(data = cor.matrix.melt, aes(x=Var1, y=Var2, fill = value)) +
  geom_tile()+
 scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = .5, limit = c(0,1), space = "Lab")
dev.off()

par(mfrow = c(1,2))
pdf(file = "Diameter and Shell Weight Scatterplot.pdf")
for (i in c(3,8)){
    plot(as.numeric(abalone_MF[,i]), as.numeric(abalone_MF[,9]),  
         xlab = paste0(col.names[i]), ylab = col.names[9],  
         main = paste0("Scatter plot ",col.names[i], " and ", col.names[9]))
}
dev.off()

par(mfrow = c(1,3))
pdf(file = "Histograms.pdf")
for (i in c(3,8,9)){
    hist(as.numeric(abalone_MF[,i]), xlab = paste0(col.names[i]), 
         main = paste0("Histogram of ",col.names[i]))
}
dev.off()

train.test.data <- function(X, portion, random = TRUE){
  n <- nrow(X)
  size <- floor(n*portion)
    train.index <- sample(c(1:n), size = size, replace = FALSE)
    X.train <- X[train.index,]
    X.test <- X[-train.index,]
  return(list(X.train = (X.train), X.test = (X.test)))
}

pdf("All Varibale Scatter Plots.pdf")
par(mfrow = c(1,2))
for (i in c(1:8)) {
  plot(as.numeric(abalone_MF[,i]), as.numeric(abalone_MF[,9]),  
       xlab = paste0(col.names[i]), ylab = col.names[9],  
       main = paste0("Scatter plot ",col.names[i], " and ", col.names[9]))
}
dev.off()

Rsqu <- function(y, ypred){
  return(1 - (sum((y - ypred)^2))/sum((y - mean(y))^2))
}

set.seed(123)
abalone_MF_train_test <- train.test.data(abalone_MF, 0.6)
abalone_MF_train <- abalone_MF_train_test$X.train
abalone_MF_test <- abalone_MF_train_test$X.test

abalone_MF_lm <- lm(Rings ~ ., data = (abalone_MF_train))
abalone_MF_fitted <- abalone_MF_lm$fitted.values

abalone_MF_train_rmse <- sqrt(mean((abalone_MF_fitted - abalone_MF_train$Rings)^2))
abalone_MF_train_r2 <- Rsqu(abalone_MF_train$Rings, abalone_MF_fitted)

abalone_MF_test_pred <- predict.lm(abalone_MF_lm, newdata = abalone_MF_test)
abalone_MF_test_rmse <- sqrt(mean((abalone_MF_test_pred - abalone_MF_test$Rings)^2))

pdf("Model Plots.pdf")
par(mfrow = c(1,3))
plot(abalone_MF_train$Rings, abalone_MF_fitted, 
     xlab = "Actual Training Values - Rings", ylab = "Fitted Values")
plot(abalone_MF_test$Rings, abalone_MF_test_pred, 
     xlab = "Acutal Test Values - Rings", ylab = "Predicted Values")
plot((abalone_MF_train$Rings - abalone_MF_fitted), xlab = "Residuals")
title("Model Prediction Plots", line = -1, outer = TRUE, cex = 1)
dev.off()

cat("For the linear regression, with all features on their original scale the in-sample Root Mean Square Error (RMSE) is:",abalone_MF_train_rmse, ".\n")
cat("For the linear regression, with all features on their original scale the test sample RMSE is:", abalone_MF_test_rmse,".\n")
cat("For the linear regression, with all features on their original scale the R-squared is:",abalone_MF_train_r2, ".\n")

normalise <- function(data){
  n <- ncol(data)
  for (i in c(1:n)){
    data[,i] <- (data[,i] - mean(data[,i]))/sd(data[,i])
  }
  return(data)
}

abalone_MF_normalised <- (normalise(abalone_MF))


set.seed(123)
abalone_MF_normalised_train_test <- train.test.data(abalone_MF_normalised, 0.6)
abalone_MF_normalised_train <- abalone_MF_normalised_train_test$X.train
abalone_MF_normalised_test <- abalone_MF_normalised_train_test$X.test

abalone_MF_normalised_lm <- lm(abalone_MF_normalised_train$Rings ~ ., 
                               data = (abalone_MF_normalised_train))
abalone_MF_normalised_fitted <- abalone_MF_normalised_lm$fitted.values

abalone_MF_normalised_train_rmse <- sqrt(mean((abalone_MF_normalised_fitted 
                                               - abalone_MF_normalised_train$Rings)^2))
abalone_MF_normalised_train_r2 <- Rsqu(abalone_MF_normalised_train$Rings,
                                       abalone_MF_normalised_fitted)

abalone_MF_normalised_test_pred <- predict.lm(abalone_MF_normalised_lm, 
                                              newdata = abalone_MF_normalised_test)
abalone_MF_normalised_test_rmse <- sqrt(mean((abalone_MF_normalised_test_pred 
                                              - abalone_MF_normalised_test$Rings)^2))

pdf("Normalised Model Plots.pdf")
par(mfrow = c(1,3))
plot(abalone_MF_normalised_train$Rings, abalone_MF_normalised_fitted, 
     xlab = "Actual Training Values - Rings", ylab = "Fitted Values")
plot(abalone_MF_normalised_test$Rings, abalone_MF_normalised_test_pred, 
     xlab = "Acutal Test Values - Rings", ylab = "Predicted Values")
plot((abalone_MF_normalised_train$Rings - abalone_MF_normalised_fitted),
     xlab = "Residuals")
title("Normalised Model Prediction Plots", line=-1, outer=TRUE)
dev.off()

abalone_MF_lm_V3V8 <- lm(abalone_MF_train$Rings ~ Diameter + `Shell weight`, 
                         data = (abalone_MF_train))
abalone_MF_fitted_V3V8 <- abalone_MF_lm_V3V8$fitted.values

abalone_MF_train_rmse_V3V8 <- sqrt(mean((abalone_MF_fitted_V3V8 - 
                                           abalone_MF_train$Rings)^2))
abalone_MF_train_r2_V3V8 <- Rsqu(abalone_MF_train$Rings, abalone_MF_fitted_V3V8)

abalone_MF_test_pred_V3V8 <- predict.lm(abalone_MF_lm_V3V8, 
                                        newdata = abalone_MF_test[,c(3,8)])
abalone_MF_test_rmse_V3V8 <- sqrt(mean((abalone_MF_test_pred_V3V8 - abalone_MF_test$Rings)^2))

pdf("Two Variable Model Plots.pdf")
par(mfrow = c(1,3))
plot(abalone_MF_train$Rings, abalone_MF_fitted_V3V8, 
     xlab = "Actual Training Values - Rings", ylab = "Fitted Values")
plot(abalone_MF_test$Rings, abalone_MF_test_pred_V3V8, 
     xlab = "Acutal Test Values - Rings", ylab = "Predicted Values")
plot((abalone_MF_train$Rings - abalone_MF_fitted_V3V8), 
     xlab = "Residuals")
title("Two Variable Model Prediction Plots", line = -1, outer = TRUE, cex = 1)
dev.off()

experiment <- function(data, portion, random = TRUE, iter){
  rmse <- r2 <- norm_rmse <- norm_r2 <- V3V8_rmse <- V3V8_r2 <- test_rmse <- norm_test_rmse <- V3V8_test_rmse <- numeric(iter)
  norm_data <- (normalise(data))
  for (i in 1:iter){
    set.seed(i)
    train_test <- train.test.data(data, 0.6)
    data_train <- train_test$X.train
    data_test <- train_test$X.test
    data_lm <- lm(Rings ~ ., data = (data_train))
    data_fitted <- data_lm$fitted.values
    rmse[i] <- sqrt(mean((data_fitted - data_train$Rings)^2))
    r2[i] <- Rsqu(data_train$Rings, data_fitted)
    data_test_pred <- predict.lm(data_lm, newdata = data_test)
    test_rmse[i] <- sqrt(mean((data_test_pred - data_test$Rings)^2))
    
    norm_train_test <- train.test.data(norm_data, 0.6)
    norm_data_train <- norm_train_test$X.train
    norm_data_test <- norm_train_test$X.test
    norm_data_lm <- lm(Rings ~ ., data = (norm_data_train))
    norm_data_fitted <- norm_data_lm$fitted.values
    norm_rmse[i] <- sqrt(mean((norm_data_fitted - norm_data_train$Rings)^2))
    norm_r2[i] <- Rsqu(norm_data_train$Rings, norm_data_fitted)
    norm_data_test_pred <- predict.lm(norm_data_lm, newdata = norm_data_test)
    norm_test_rmse[i] <- sqrt(mean((norm_data_test_pred - norm_data_test$Rings)^2))
    
    V3V8_data_lm <- lm(Rings ~ Diameter + `Shell weight` , data = (data_train))
    V3V8_data_fitted <- V3V8_data_lm$fitted.values
    V3V8_rmse[i] <- sqrt(mean((V3V8_data_fitted - data_train$Rings)^2))
    V3V8_r2[i] <- Rsqu(data_train$Rings, V3V8_data_fitted)
    V3V8_data_test_pred <- predict.lm(V3V8_data_lm, newdata = data_test)
    V3V8_test_rmse[i] <- sqrt(mean((V3V8_data_test_pred - data_test$Rings)^2))
  }
  return(list(RMSE = rmse, Test.RMSE = test_rmse, R2 = r2, Norm.RMSE = norm_rmse, 
              Norm.Test.RMSE = norm_test_rmse, Norm.R2 = norm_r2, 
              V3V8.RMSE = V3V8_rmse, V3V8.Test.RMSE = V3V8_test_rmse, 
              V3V8.R2 = V3V8_r2))
}

exper <- experiment(abalone_MF, 0.6, random = T, iter = 30)
cat("Linear regression, with all features on their original scale:\n")
cat("For the linear regression, with all features on their original scale the 
    mean and standard deviation of the in-sample Root Mean Square Error (RMSE) is:",
    mean(exper$RMSE), ",", sd(exper$RMSE), ".\n")
cat("For the linear regression, with all features on their original scale the 
    mean and standard deviation of the test sample RMSE is:", mean(exper$Test.RMSE),
    ",",sd(exper$Test.RMSE),".\n")
cat("For the linear regression, with all features on their original scale the
    mean and standard deviation of the R-squared is:", mean(exper$R2), ",", 
    sd(exper$R2), ".\n")

cat("\n Linear regression, with all input data normalised:\n")
cat("For the linear regression, with all input data normalised, the mean and 
    standard deviation of the in-sample RMSE is:", mean(exper$Norm.RMSE), ",", 
    sd(exper$Norm.RMSE),".\n")
cat("For the linear regression, with all input data normalised, the mean and 
    standard deviation of the test sample RMSE is:", mean(exper$Norm.Test.RMSE), 
    ",", sd(exper$Norm.Test.RMSE), ".\n")
cat("For the linear regression, with all input data normalised, the mean and 
    standard deviation of the R-squared is:", mean(exper$Norm.R2), ",", 
    sd(exper$Norm.R2) , ".\n")

cat("\n Linear regression, with only Diamater and Shell Weight as features:\n")
cat("For the linear regression, with only Diamater and Shell Weight as features,
    the mean and standard deviation of the in-sample RMSE is:", mean(exper$V3V8.RMSE), 
    ",",sd(exper$V3V8.RMSE),".\n")
cat("For the linear regression, with only Diamater and Shell Weight as features,
    the mean and standard deviation of the test sample RMSE is:", 
    mean(exper$V3V8.Test.RMSE),",", sd(exper$V3V8.Test.RMSE), ".\n")
cat("For the linear regression, with only Diamater and Shell Weight as features,
    the mean and standard deviation of the R-squared is:", mean(exper$V3V8.R2), 
    ",", sd(exper$V3V8.R2), ".\n")

    
