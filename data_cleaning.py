#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:41:04 2020

@author: yangzhang
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#handling missing values - training data
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_df = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
vars_wt_missing = missing_df[missing_df['Total'] > 0]
print(vars_wt_missing.shape)
vars_wt_missing

##drop features with > 40% missing values
df_train = df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)

##replace missing values with NA or median
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].dropna().median())
df_train['GarageCond'] = df_train['GarageCond'].fillna('NA')
df_train['GarageType'] = df_train['GarageType'].fillna('NA')
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna('NA')
df_train['GarageFinish'] = df_train['GarageFinish'].fillna('NA')
df_train['GarageQual'] = df_train['GarageQual'].fillna('NA')
df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna('NA')
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna('NA')
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna('NA')
df_train['BsmtCond'] = df_train['BsmtCond'].fillna('NA')
df_train['BsmtQual'] = df_train['BsmtQual'].fillna('NA')
df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna('NA')
df_train['MasVnrType'] = df_train['MasVnrType'].fillna('NA')
df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].dropna().sort_values().index[0])

##checking missing values again
missing=df_train.isnull().sum().sort_values(ascending = False)
missing=missing.drop(missing[missing==0].index)
print(missing.shape)


#handling missing values - testing data
total = df_test.isnull().sum().sort_values(ascending = False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending = False)
missing_df = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
vars_wt_missing = missing_df[missing_df['Total'] > 0]
print(vars_wt_missing.shape)
vars_wt_missing

##drop features with > 40% missing values
df_test = df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)

##replace missing values with NA or median
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].dropna().median())
df_test['GarageCond'] = df_test['GarageCond'].fillna('NA')
df_test['GarageType'] = df_test['GarageType'].fillna('NA')
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna('NA')
df_test['GarageFinish'] = df_test['GarageFinish'].fillna('NA')
df_test['GarageQual'] = df_test['GarageQual'].fillna('NA')
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna('NA')
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna('NA')
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna('NA')
df_test['BsmtCond'] = df_test['BsmtCond'].fillna('NA')
df_test['BsmtQual'] = df_test['BsmtQual'].fillna('NA')
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna('NA')
df_test['MasVnrType'] = df_test['MasVnrType'].fillna('NA')
df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].dropna().sort_values().index[0])
df_test['GarageCars'] = df_test['GarageCars'].fillna('NA')
df_test['GarageArea'] = df_test['GarageArea'].fillna('NA')
df_test['KitchenQual'] = df_test['KitchenQual'].fillna('NA')
df_test['Exterior1st'] = df_test['Exterior1st'].fillna('NA')
df_test['SaleType'] = df_test['SaleType'].fillna('NA')

df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna('NA')
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna('NA')
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna('NA')
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna('NA')
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna('NA')
df_test['Utilities'] = df_test['Utilities'].fillna('NA')
df_test['Functional'] = df_test['Functional'].fillna('NA')
df_test['MSZoning'] = df_test['MSZoning'].fillna('NA')
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna('NA')
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna('NA')

##checking missing values again
missing=df_test.isnull().sum().sort_values(ascending = False)
missing=missing.drop(missing[missing==0].index)
missing.shape

#export dataframes to csvs
df_train.to_csv('train_clean.csv', index = False)
df_test.to_csv('test_clean.csv', index = False)
