#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:25:47 2020

@author: yangzhang
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('train_transformed.csv')
df = df.drop(['GarageAge', 'MasVnrArea'], axis = 1) ## need to check why there are Nan in those two variables after cleaning.

###############
### Scaling
#############

#Split into train-validation set
y = df.SalePrice
train_X, val_X, train_y, val_y = train_test_split(df, y, random_state = 0)


#####################
### regression model
######################
lr_model=LinearRegression()
lr_model.fit(train_X,train_y)

Y_pred_lr=lr_model.predict(val_X)
Y_lr_train=lr_model.predict(train_X)

##Evaluation the model trhough different metrics
#Root Mean Sqaured Error
#Evaludating the model

MSE = np.sqrt(mean_squared_error(train_y,Y_lr_train))
print("RMSE of the Linear Regression Model is :",np.sqrt(MSE))

r2_score=lr_model.score(train_X,train_y)
print("R2_score of Liner Regression",r2_score)

#Lets check Adjusted R_Squared 
adj_r2 = (1 - (1 - r2_score) * ((train_X.shape[0] - 1) / 
          (train_X.shape[0] - train_X.shape[1] - 1)))
print("Adjusted R_Sqaured of Linear Regression Model is :",adj_r2 )


df_test = pd.read_csv("test_clean.csv")
df_test = df_test.drop(['GarageAge', 'MasVnrArea'], axis = 1)

features = ['LotArea', 'OverallQual', 'GrLivArea']
test_X = df_test[features]
test_preds = lr_model.predict(test_X)

#Submission
output = pd.DataFrame({'Id': df_test.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)