# 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
# Importing the dataset
v1 = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\LassoRidge\\50_Startups.csv")
# Rearrange the order of the variables
v1 = v1.iloc[:, [1, 0, 2, 3, 4]]
v1.columns
# Correlation matrix 
a = v1.corr()
a
# EDA
a1 = v1.describe()
# Sctter plot and histogram between variables
sns.pairplot(v1) # sp-hp, wt-vol multicolinearity issue
# Preparing the model on train data 
model_train = smf.ols("Profit ~ AD + RD + MS", data = v1).fit()
model_train.summary()
# Prediction
pred = model_train.predict(v1)
# Error
resid  = pred - v1.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
v1 = v1.drop(["State"],axis=1)
# To overcome the issues, LASSO and RIDGE regression are used

#LASSO MODEL
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13, normalize = True)
lasso.fit(v1.iloc[:, 1:], v1.Profit)
# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(v1.columns[1:]))
lasso.alpha
pred_lasso = lasso.predict(v1.iloc[:, 1:])
# Adjusted r-square
lasso.score(v1.iloc[:, 1:], v1.Profit)
# RMSE
np.sqrt(np.mean((pred_lasso - v1.Profit)**2))

# RIDGE REGRESSION 
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)
rm.fit(v1.iloc[:, 1:], v1.Profit)
# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_
plt.bar(height = pd.Series(rm.coef_), x = pd.Series(v1.columns[1:]))
rm.alpha
pred_rm = rm.predict(v1.iloc[:, 1:])
# Adjusted r-square
rm.score(v1.iloc[:, 1:], v1.Profit)
# RMSE
np.sqrt(np.mean((pred_rm - v1.Profit)**2))

# ELASTIC NET REGRESSION 
from sklearn.linear_model import ElasticNet 
enet = ElasticNet(alpha = 0.4)
enet.fit(v1.iloc[:, 1:], v1.Profit) 
# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_
plt.bar(height = pd.Series(enet.coef_), x = pd.Series(v1.columns[1:]))
enet.alpha
pred_enet = enet.predict(v1.iloc[:, 1:])
# Adjusted r-square
enet.score(v1.iloc[:, 1:], v1.Profit)
# RMSE
np.sqrt(np.mean((pred_enet - v1.Profit)**2))

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(v1.iloc[:, 1:], v1.Profit)
lasso_reg.best_params_
lasso_reg.best_score_
lasso_pred = lasso_reg.predict(v1.iloc[:, 1:])
# Adjusted r-square#
lasso_reg.score(ash.iloc[:, 1:], v1.Profit
# RMSE
np.sqrt(np.mean((lasso_pred-v1.Profit)))

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(v1.iloc[:, 1:], v1.Profit)
ridge_reg.best_params_
ridge_reg.best_score_
ridge_pred = ridge_reg.predict(v1.iloc[:, 1:])
# Adjusted r-square#
ridge_reg.score(v1.iloc[:, 1:], v1.Profit)
# RMSE
np.sqrt(np.mean((ridge_pred - v1.Profit)**2))

# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
enet = ElasticNet()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(v1.iloc[:, 1:], v1.Profit)
enet_reg.best_params_
enet_reg.best_score_
enet_pred = enet_reg.predict(v1.iloc[:, 1:])
# Adjusted r-square
enet_reg.score(v1.iloc[:, 1:], v1.Profit)
# RMSE
np.sqrt(np.mean((enet_pred - v1.Profit)**2))
