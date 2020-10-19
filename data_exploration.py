#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:47:40 2020

@author: yangzhang
"""

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style(style='white')
sns.set(rc={'figure.figsize':(10,6)})
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab

df = pd.read_csv('train_clean.csv')

####################################
######## Data Visualization ########
####################################
print("Find most important features relative to target")
corr=df.drop('Id', axis = 1).corr().sort_values(by='SalePrice',ascending=False).round(2)
print(corr['SalePrice'])

#deal with outliers
#GrLivArea has the highest correlation with SalePrice, let's look at that first.
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#looks like the two points at far right are outliers, remove both.
#deleting points
df.sort_values(by = 'GrLivArea', ascending = False)[:2]
df = df.drop(df[df['Id'] == 1299].index)
df = df.drop(df[df['Id'] == 524].index)

#target variable distribution
plt.figure(figsize=(10,6))
plt.title("Before transformation of SalePrice")
dist = sns.distplot(df['SalePrice'],norm_hist=False)
stats.probplot(df['SalePrice'], dist='norm', plot=pylab)
#as we can see, the target variable is right-skewed, needs to be log transformed.
plt.title("After transformation of SalePrice")
sns.distplot(np.log(df['SalePrice']),norm_hist=False)
stats.probplot(df['SalePrice'], dist='norm', plot=pylab)

#transform SalePrice
df["SalePrice"] = np.log(df['SalePrice'])


#####################################
######## Feature engineering ########
#####################################
##1.how old the house when it was sold
df['Age'] = df['YrSold'] - df['YearBuilt']

##2.remodeled, yes or no?
df['Remodeled'] = 0
df.loc[df['YearBuilt'] != df['YearRemodAdd'], 'Remodeled'] = 1
df = pd.get_dummies(df, columns=['Remodeled'])

##3.how old was the garage?
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

##4.total number of baths in the house
df['TotBaths'] = df['BsmtFullBath'] + df['FullBath'] + df['BsmtHalfBath'] * 0.5 + df['HalfBath'] * 0.5

##5.total floor feet
df['Floorfeet'] = df['1stFlrSF'] + df['2ndFlrSF']

##6.total finished area in basement
df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']

##7.pct bedroom
df['pctBedroomAbvGr'] = df['BedroomAbvGr'] / df['TotRmsAbvGrd']

##8.total lot area
df['TotalLot'] = df['LotFrontage'] + df['LotArea']

##9.total porch area
df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']

# drop columns
drp_cols = ['YearRemodAdd', 'YrSold', 'YearBuilt', 'GarageYrBlt', \
            'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', \
            '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', \
            'PoolArea', 'MoSold', 'MiscVal', 'Floorfeet', 'Fireplaces', \
            'LotFrontage', 'LotArea', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
df = df.drop(drp_cols, axis = 1)

## Converting Categorical to Numerical
df = pd.get_dummies(df)

df.to_csv('train_transformed.csv', index=False)














