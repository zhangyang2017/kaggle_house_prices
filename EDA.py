#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:08:02 2020

@author: yangzhang
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

## Realized i need to clean both training data set and testing data set together. So starting over.
#import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print('Data Description')
print('The shape of training set: ', df_train.shape[0], 'rows ', 'and', df_train.shape[1]  , 'columns'  )
print('The shape of testing set: ', df_test.shape[0], 'rows', 'and', df_test.shape[1], 'columns')

#Here's one way to combine train and test data then split them up after data processing
#resource: https://www.kaggle.com/vishalvanpariya/top-5-on-leaderboard
trainrow = df_train.shape[0]
testrow = df_test.shape[0]
testids = df_test['Id'].copy()
y_train = df_train['SalePrice'].copy()

X = pd.concat((df_train, df_test)).reset_index(drop=True)
X = X.drop('SalePrice', axis = 1)
X = X.drop('Id', axis = 1)

#handling missing values - X
missing_vals = X.isnull().sum().sort_values(ascending = False)
percent = ( X.isnull().sum()/X.isnull().count() ).sort_values(ascending = False)
missing_df = pd.concat([missing_vals, percent], axis = 1, keys = ["Total", "Percent"])
total_missing = missing_df[missing_df['Total'] > 0]
print(total_missing.shape)
total_missing

##drop features with > 40% missing values in both data sets
X = X.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)

#Other categorical and numerial features has nearly Zero percent missing values Imputing with mode and median.
##resource: https://www.kaggle.com/mhrizvi/simple-code-top-30-in-competition
num_features_list = X.select_dtypes(include=np.number).drop(['MSSubClass'], axis = 1).columns.tolist() #MSSubClass is nominal
cat_features_list = X.select_dtypes(exclude=np.number).columns.tolist()
cat_features_list.append('MSSubClass')

#Categorical Features, replace missing values with mode
for feature in cat_features_list:
    X[feature].fillna(X[feature].mode()[0], inplace=True)

#Numerical Features, replace missing values with median
for feature in num_features_list:
    X[feature].fillna(X[feature].median(), inplace=True)

#Ensuring all missing value are handled properly
X.isnull().sum().sort_values(ascending=False).head(50)


#EDA
##visualization

##split data into numerical and categorical for data visualization
numeric_ = X[num_features_list]
cat_ = X[cat_features_list]  
cat_.columns
numeric_.columns

##Univariate Analysis
##check target variable distribution SalePrice
sns.distplot(y_train);
fig = plt.figure()
res = stats.probplot(y_train, plot=plt)
##SalePrice shows right skewness, need log transform later.

##check distributions for numeric variables
fig = plt.figure(figsize=(15,18))
for index, col in enumerate(numeric_.columns):
    plt.subplot(5, 7, index+1)
    sns.distplot(numeric_.loc[:,col].dropna(), kde=False)
fig.tight_layout(pad=1.0)
## variables with mostly one value that I probably will just drop
## BsmtFinSF2, LowQualFinSF, EnclosedPorch, 35snPorch, ScreenPorch, PoolArea, MiscVal

fig = plt.figure(figsize=(14,15))
for index,col in enumerate(numeric_.columns):
    plt.subplot(5,7,index+1)
    sns.boxplot(y=col, data=numeric_.dropna())
fig.tight_layout(pad=1.0)

##for catagorical variables
fig = plt.figure(figsize=(18,20))
for index in range(len(cat_.columns)):
    plt.subplot(8,5,index+1)
    sns.countplot(x=cat_.iloc[:,index], data=cat_.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)


##Bivariate Analysis
plt.figure(figsize=(14,12))
correlation = numeric_.corr()
sns.heatmap(correlation, mask = correlation < 0.7, linewidth = 0.5, cmap='Blues')

fig = plt.figure(figsize=(20,20))
for index in range(len(numeric_.columns)):
    plt.subplot(10,5,index+1)
    sns.scatterplot(x=numeric_.iloc[:,index], y=y_train, data=numeric_.dropna())
fig.tight_layout(pad=1.0)
##variables need to drop (useless features in predicting saleprice): 
##LowQualFinSF, EnclosedPorch, 35snPorch, ScreenPorch, PoolArea, MiscVal, MoSold
##new features: total bathrooms, house age, garage age, total area


##Analyses of features
##resource:https://www.kaggle.com/drsergio/visualization-feature-generation-house-prices/data

##1. MSSubClass: Identifies the type of dwelling involved in the sale
fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('MSSubClass').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("MSSubClass - type of dwelling involved in the sale",size=13)
plt.tight_layout()

X_train = X.iloc[:trainrow]
X_train['SalePrice'] = pd.Series(y_train)
sns.boxplot('MSSubClass', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS MSSubClass",size=13)
plt.tight_layout()


#lets split data using trainrow data and scale data

X_train=X.iloc[:trainrow]
X_test=X.iloc[trainrow:]



scaler=StandardScaler()
scaler=scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
