# 4
library(readr)
# Importing the dataset
v1 <- read.csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\LassoRidge\\Life_expectencey_LR.csv",stringsAsFactors = TRUE)
# Reorder the variables
v1 <- v1[,c(2,1,3,4,5)]
v1[is.na(v1)==TRUE]<-0
library(glmnet)
x <- model.matrix(Life_expectancy ~ ., data = v1)[,-1]
y <- v1$Life_expectancy
grid <- 10^seq(10, -2, length = 110)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)

# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)
cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min
y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared
predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

