#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:25:47 2020

@author: yangzhang
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler

df_train = pd.read_csv('train_cleaned.csv')
df_test = pd.read_csv('test_cleaned.csv')

###final check before modeling
##correlation matrix to check redundant features
plt.figure(figsize=(14,12))
correlation = df_train.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')

df_train = df_train.drop(['LotArea', 'GrLivArea'], axis = 1)
df_test = df_test.drop(['LotArea', 'GrLivArea'], axis = 1)

df_train = pd.get_dummies(df_train)
df_train = df_train.drop(['Condition2_RRAe', 'Condition2_RRAn', \
              'Condition2_RRNn', 'HouseStyle_2.5Fin'], axis = 1)
df_test = pd.get_dummies(df_test)
##log transform SalePrice
df_train["SalePrice"] = np.log(df_train['SalePrice'])

#Split train data into train-validation set
y = df_train.SalePrice
X = df_train.drop(['SalePrice'], axis = 1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

##use RobustScaler to scale our data because it's powerful against outliers
scaler = scaler= RobustScaler()
# transform "train_X"
train_X = scaler.fit_transform(train_X)
# transform "val_X"
val_X = scaler.transform(val_X)

X = scaler.fit_transform(X)

#Transform the test set
df_test_scaled = scaler.fit_transform(df_test)
####################################################
##################### Modeling #####################
####################################################

############################
### 1. linear regression ###
############################
lreg = LinearRegression()
lreg.fit(train_X, train_y)
Y_pred_lr=lreg.predict(val_X)

##Evaluation the model through different metrics
#Root Mean Sqaured Error
MSE=np.sqrt(mean_squared_error(val_y,Y_pred_lr))
print("RMSE of the Linear Regression Model is :",np.sqrt(MSE))

test_preds = lreg.predict(df_test_scaled)
#Submission
output = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': np.exp(test_preds)})
output.to_csv('submission.csv', index=False)
##Your submission scored 16822.51975

##############################
######### 2. XGBoost #########
##############################

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(train_X, train_y)], 
             verbose=False)
print(my_model.score(train_X,train_y))

Y_xgb_train = my_model.predict(train_X)
print("RMSE of XgBosst Model is",np.sqrt(mean_squared_error(train_y, Y_xgb_train)))

test_preds = my_model.predict(df_test_scaled)
#Submission
output = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': np.exp(test_preds)})
output.to_csv('submission.csv', index=False)
### 15536.57624, best so far.

##if i train with all the train data
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X, y, 
             early_stopping_rounds=5, 
             eval_set=[(X, y)], 
             verbose=False)
print(my_model.score(X,y))

test_preds = my_model.predict(df_test_scaled)
#Submission
output = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': np.exp(test_preds)})
output.to_csv('submission.csv', index=False)

##XGBoost HyperParameter Tuning
from sklearn.model_selection import RandomizedSearchCV

##https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning#4.-Modeling
param_lst = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators' : [100, 500, 1000, 2000, 3000],
    'max_depth' : [3, 6, 9],
    'min_child_weight' : [1, 5, 10, 20],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1]
}

xgb_reg = RandomizedSearchCV(estimator = my_model, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)
       
xgb_search = xgb_reg.fit(X, y)

# XGB with tune hyperparameters
best_param = xgb_search.best_params_
xgb = XGBRegressor(**best_param)
print(xgb.score(X,y))

test_preds = xgb.predict(df_test_scaled)
#Submission
output = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': np.exp(test_preds)})
output.to_csv('submission.csv', index=False)
## no better than not tuning


##############################
######### random forest #########
##############################
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=500)
rfr.fit(train_X,train_y)
print(rfr.score(train_X,train_y))

test_preds = rfr.predict(df_test_scaled)
#Submission
output = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': np.exp(test_preds)})
output.to_csv('submission.csv', index=False)
##17311.19703