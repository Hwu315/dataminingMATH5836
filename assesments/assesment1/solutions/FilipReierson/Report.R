#' ---
#' title: "Abalone dataset: Predicting the Ring Age in Years"
#' author: "Filip Reierson"
#' date: "25/09/2021"
#' output: pdf_document
#' ---
#' 
#' \tableofcontents
#' 
## ----setup, include=FALSE--------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message=FALSE)

#' 
#' 
#' ## Data processing
#' The nine attributes of the Abalone dataset.
#' 
#' | Name |	Data Type	| Meas.	| Description |
#' | ----	|	--------- |	----- |	----------- |
#' | Sex	|	nominal	| -- | M, F, and I (infant) |
#' | Length	|	continuous |	mm |	Longest shell measurement |
#' | Diameter |	continuous |	mm |	perpendicular to length |
#' | Height	|	continuous |	mm |	with meat in shell |
#' | Whole weight |	continuous |	grams |	whole abalone |
#' | Shucked weight |	continuous |	grams |	weight of meat |
#' | Viscera weight |	continuous |	grams |	gut weight (after bleeding) |
#' | Shell weight |	continuous |	grams |	after being dried |
#' | Rings	|	integer |	-- |	+1.5 gives the age in years |
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
library(tidyverse)
library(knitr)
library(cowplot)
column_names <- c('Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings')
abalone <- read_csv('data/abalone.data', col_names = F, col_types = list(X9=col_integer()))
names(abalone) <- column_names

abalone <- abalone %>%
  mutate(Sex = case_when(Sex == 'I' ~ -1,
                         Sex == 'M' ~ 0,
                         Sex == 'F' ~ 1))

#' 
#' We begin by replacing Sex values I, M, and F by -1, 0, and 1 respectively.
#' 
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
cormap <- as_tibble(cor(abalone), rownames='Var1') %>%
  pivot_longer(-Var1, names_to = 'Var2', values_to = 'r')
   
cormap %>%
  ggplot(aes(Var1, Var2, fill=r, label=round(r,2))) + 
    geom_tile() + 
    geom_text(color='white') +
    labs(x='',y='', fill="Pearson's r") +
    scale_fill_gradient2(midpoint = 0, limit = c(-1,1)) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

#' 
#' On the correlation map we can observe that correlation between rings and other features ranges from 0.4 to 0.63 all in the positive direction. The sex correlation can't be meaningfully interpreted here as it is a nominal variable. One hot encoding, would be appropriate, but is outside the scope. We can also see that the various measures of weight are strongly correlated, as we might expect. We can also see that Length and Diameter is very highly correlated, and as a result would make coefficient interpretation problematic if they are both included in the model. We can also read off the correlation plot which features are most correlated with ring-age.  
#' 
#' The features most correlated with ring-age are,
## --------------------------------------------------------------------------------------------------------------------------------------
most_cor <- cormap %>%
  filter(Var1=='Rings',Var2!='Rings') %>%
  arrange(desc(r)) %>%
  top_n(2,r) %>%
  select(Var2, r) %>%
  rename(Feature = Var2)
most_cor %>%
  kable(digits=2)

#' 
## ----fig.height=4, fig.width=8---------------------------------------------------------------------------------------------------------
p1 <- abalone %>%
  ggplot(aes(`Shell weight`, Rings)) +
  geom_point()
p2 <- abalone %>%
  ggplot(aes(Diameter, Rings)) +
  geom_point()
plot_grid(p1, p2)

#' 
#' In the above plot we can see that shell weight doesn't appear to have a strictly linear relationship with rings, but there is clearly a strong association. Diameter appears to have a more linear looking relationship for low values, but has a concave up shape for later values. Both relationships appear to have fairly large heteroscedasticity in the form of fanning, suggesting estimations of ring age may worsen as number of rings increases.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
p1 <- abalone %>%
  ggplot(aes(Rings)) +
  geom_histogram(binwidth = 1, fill='steelblue', color='black')
p2 <- abalone %>%
  ggplot(aes(`Shell weight`)) +
  geom_histogram(bins = 50, fill='steelblue', color='black')
p3 <- abalone %>%
  ggplot(aes(Diameter)) +
  geom_histogram(bins = 50, fill='steelblue', color='black')
plot_grid(p1, p2, p3)

#' 
#' Rings appear to be fairly symmetrically distributed, perhaps with a small right skew. Shell weight is right skewed, and may be bimodal, although it is not entirely clear from this plot. Diameter is highly left skewed.
#' 
#' We can confirm shell weight is bimodal with the following stacked histogram.
#' 
## ----fig.height=4, fig.width=8---------------------------------------------------------------------------------------------------------
abalone %>%
  ggplot(aes(`Shell weight`, fill=as.factor(Sex))) +
  geom_histogram(bins = 50, position = 'fill', color='black') +
  labs(fill='Sex', y = 'proportion')

#' A better model may use sex in the model, perhaps with an interaction effect, however, this is outside the report's scope. 
#' 
#' With that said I will move on to modelling. 
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
train_test_split <- function(df, prop, seed) {
  set.seed(seed)
  train <- sample(1:nrow(df), round(prop*nrow(df)), replace=F)
  df_train <- df[train,]
  df_test <- df[-train,]
  return(list('train' = df_train, 'test' = df_test))
}

normalise <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

normalise_abalone <- function(df) {
  df_scaled <- df %>%
    mutate(across(-c(Sex, Rings),normalise))
  return(df_scaled)
}

normalised_abalone <- normalise_abalone(abalone)
abalone_split_normalised <- train_test_split(normalised_abalone, 0.6, 0)
abalone_split <- train_test_split(abalone, 0.6, 0)

#' 
#' 
#' ## Modelling  
## --------------------------------------------------------------------------------------------------------------------------------------
model <- lm(Rings ~ ., data=abalone_split$train)
model_normalised <- lm(Rings ~ ., data=abalone_split_normalised$train)

model_plots <- function(model, data) {
  df <- data.frame(Predicted = predict(model, data),
           Actual = data$Rings) %>%
    mutate(Residual = scale(Actual-Predicted))
  p1 <- ggplot(df, aes(Predicted, Actual)) +
    geom_point() +
    geom_abline(slope=1,intercept=0, linetype='dashed')
  p2 <- ggplot(df, aes(Residual)) +
    geom_histogram(bins = 30, fill='steelblue', color='black')
  p3 <- ggplot(df, aes(sample=Residual)) +
    geom_qq() +
    geom_qq_line() +
    xlab('Theoretical Quantiles') + 
    ylab('Standardised Residuals')
  return(list('Prediction' = p1, 'Residuals' = p2, 'QQ' = p3))
}

model_plots_train_test <- function(model, abalone_split) {
  p_train <- model_plots(model, abalone_split$train)
  p_test <- model_plots(model, abalone_split$test)
  plot_grid(p_train$Prediction, p_train$Residual, p_train$QQ,
            p_test$Prediction, p_test$Residuals, p_test$QQ,
            labels = c('Train', 'Train', 'Train', 
                       'Test', 'Test', 'Test'), 
            label_x = .1, label_y = .95, nrow = 2)
}

get_summary_stats <- function(y_hat, y) {
  return(data.frame(list(
    'R squared' = cor(y, y_hat) ^ 2,
    'RMSE' = sqrt(mean((y - y_hat)^2))
  ), check.names = F))
}

summarise_train_test <- function(model, abalone_split) {
  rbind('Train' = get_summary_stats(model$fitted.values, abalone_split$train$Rings),
        'Test' = get_summary_stats(predict(model, abalone_split$test), abalone_split$test$Rings)) %>%
    rownames_to_column('Set')
}

#' 
#' ### Full linear model non-normalised
#' Fitting the full model gives coefficients:
## --------------------------------------------------------------------------------------------------------------------------------------
broom::tidy(model) %>%
  kable(digits=4)

#' 
#' We can see that Diameter and Length explain much of the same effect as expected from the correlation plot. Also note that Sex is not one hot encoded, so interpretation is difficult. 
#' Length, height, and diameter are all associated with ring-years. Different measures of weight have different associations with ring-years when considering all features. Whole weight and shell weight are positively associated with ring-years, while shucked weight and viscera weight are negatively associated with ring-years.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
model_plots_train_test(model, abalone_split)

#' 
#' The actual vs predicted plot indicate that a linear model isn't great, but explains some of the variation in ring-years. The qq-plot indicate that normality of residuals is not an appropriate assumption. However, since this is a predictive exercise rather than explanatory it is not problematic as long as the model has good predictive power on the test set. However, looking at the test set we see that at least one residual is way out and in general the estimates are poor for higher quantiles.  
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
summarise_train_test(model, abalone_split) %>%
  kable(digits = 3)

#' 
#' The $R^2$ value indicates that 51.4\% of the ring-year's variability in the test set was explained by the model. 
#' 
#' ### Full model normalised
## --------------------------------------------------------------------------------------------------------------------------------------
broom::tidy(model_normalised) %>%
  kable(digits=4)

#' 
## --------------------------------------------------------------------------------------------------------------------------------------
summarise_train_test(model_normalised, abalone_split_normalised) %>%
  kable(digits = 3)

#' 
#' We observe that while the normalisation changes the coefficients of our model it does not affect its ability to predict. This is because ordinary least squares is scale invariant, so a linear transformation doesn't impact it. 
#' 
#' 
#' ### Linear model two features
#' Recall we found that shell weight and diameter had the highest correlation with rings, so we will develop a model using only those features as predictors.
## --------------------------------------------------------------------------------------------------------------------------------------
model2 <- lm(Rings ~ `Shell weight`+`Diameter`, data=abalone_split$train)
model2_normalised <- lm(Rings ~ `Shell weight`+`Diameter`, data=abalone_split_normalised$train)

#' 
## --------------------------------------------------------------------------------------------------------------------------------------
model_plots_train_test(model2, abalone_split)

#' 
## --------------------------------------------------------------------------------------------------------------------------------------
broom::tidy(model2) %>%
  kable(digits=4)

#' 
## --------------------------------------------------------------------------------------------------------------------------------------
summarise_train_test(model2, abalone_split_normalised) %>%
  kable(digits = 3)

#' 
#' While the metrics indicate a worse fit than the full model it is worth noting that this model appears to generalise better to the test set. Looking at the test set qq-plots we don't see the same deviation from normality for lower quantiles as we saw in the full model. However, Diameter doesn't add meaningfully to the model as indicated by the p-value, so a more parsimonious model wouldn't include diameter. 
#' 
#' 
## --------------------------------------------------------------------------------------------------------------------------------------
broom::tidy(model2_normalised) %>%
  kable(digits=4)

#' 
## --------------------------------------------------------------------------------------------------------------------------------------
summarise_train_test(model2_normalised, abalone_split_normalised) %>%
  kable(digits = 3)

#' 
#' As before, normalising doesn't affect our metrics. 
#' 
#' ### Sensitivity analysis
## --------------------------------------------------------------------------------------------------------------------------------------
model1_experiment <- function(abalone, seed) {
  df_split <- train_test_split(abalone, .6, seed)
  model <- lm(Rings~., df_split$train)
  outcome <- summarise_train_test(model, df_split)
  return(outcome)
}

model2_experiment <- function(abalone, seed) {
  df_split <- train_test_split(abalone, .6, seed)
  model <- lm(Rings ~ `Shell weight`+`Diameter`, df_split$train)
  outcome <- summarise_train_test(model, df_split)
  return(outcome)
}

run_experiment <- function(abalone, experiment) {
  outcomes <- list()
  for (i in 1:30) {
    outcomes[[i]] <- experiment(abalone, i)
  }
  result <- Reduce(rbind, outcomes) %>%
    pivot_longer(-Set, names_to = 'Statistic') %>%
    group_by(Set, Statistic) %>%
    summarise(Mean = mean(value),
              SD = sd(value))
  return(result)
}

#' 
## --------------------------------------------------------------------------------------------------------------------------------------
full_model_non_normalised <- run_experiment(abalone, model1_experiment) 
full_model_normalised <- run_experiment(normalised_abalone, model1_experiment)
reduced_model_non_normalised <- run_experiment(abalone, model2_experiment)
reduced_model_normalised <- run_experiment(normalised_abalone, model2_experiment)
table_list <- lapply(list(full_model_non_normalised,full_model_normalised, 
                                 reduced_model_non_normalised, reduced_model_normalised), kable, digits=3)

#' 
#' The follow table shows aggregate statistics from the full model being fitted 30 times with randomised splits. 
## --------------------------------------------------------------------------------------------------------------------------------------
table_list[[1]]

#' 
#' And the following shows the aggregate statistics from the full model being fitted 30 times in the same manner, but with the normalised inputs.
## --------------------------------------------------------------------------------------------------------------------------------------
table_list[[2]]


#' 
#' Again we observe that the metrics are identical between normalised and non-normalised, due to the invariance properties of the estimates. We would observe a different result if we didn't use the same seeds for the experiments (1 to 30).
#' 
#' The following table was calculated by running the two feature model on 30 random 60/40 splits. 
## --------------------------------------------------------------------------------------------------------------------------------------
table_list[[3]]

#' Here we see that the standard deviation of both $R^2$ and RMSE are marginally lower for the test set, while standard deviation of R squared in the training set is almost identical in both models. The training set RMSE is marginally higher in the two feature model. The two feature model also maintains its R squared on the test set while the full model's R squared is marginally lower on the test data, this suggests the simpler model generalises slightly better. The full model explains about 13\% more of the variation in ring-years than the two feature model.
#' 
#' 
#' The follow table shows aggregate statistics from the two feature model using normalised inputs being fitted 30 times with randomised splits. 
## --------------------------------------------------------------------------------------------------------------------------------------
table_list[[4]]

#' 
#' Again we observe the invariance property.
#' 
#' 
#' ## Conclusion
#' Neither model appears appears to be ideal. What I would look at for future analyses:
#' 
#' 1. Using one hot encoding to deal with Sex (since there are 3 categories)
#' 2. Interaction effects.
#' 3. Transformations that better deal with the non-linear residuals. 
#' 4. Hierarchical clustering by variable to help select better features for the model.
#' 
#' ## References
#' Other than the R packages imported I did not use any existing code. R packages used: tidyverse, knitr, cowplot, and broom.
#' 
#' 
#' 
