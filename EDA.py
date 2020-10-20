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
trainids = df_train['Id'].copy()
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
##conclusions:
##(1) the type of house most sold was one-story and was built after 1946
##(2) newer houses were sold more than older ones
##(3) two-story houses sold more than one-story ones
##(4) no clear correlation between SalePrice is observed
## there's a way to code it like this:
## qual_dict = {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 
             ##80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 1, 180: 0, 190: 0}
##df['MSSubClass'] = df['MSSubClass'].map(qual_dict).astype(object)
##del qual_dict
## to differentiate newer models from older ones. but i think other variables have this information already.
## drop this.

##2. MSZoning: Identifies the general zoning classification of the sale
fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('MSZoning').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("MSZoning - general zoning classification of the sale",size=13)
plt.tight_layout()

sns.boxplot('MSZoning', 'SalePrice', data = X_train, order=['RL', 'RM', 'FV', 'RH', 'C (all)'], \
            ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS MSZoning",size=13)
plt.tight_layout()
##conclusions:
    ##(1) most houses sold were in low and medium population density area
    ##not particularly important info, drop this.

##3. Street: Type of road access to property
print (X.groupby('Street').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('Street').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Type of road access to property",size=13)
plt.tight_layout()

sns.boxplot('Street', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS Street",size=13)
plt.tight_layout()
##conclusion: very few houses have Grvl pavement, so drop it


##4. LotShape: General shape of the property [drop it]
fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('LotShape').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("General shape of the property",size=13)
plt.tight_layout()

sns.boxplot('LotShape', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS LotShape",size=13)
plt.tight_layout()

##5. Land Contour: Flatness of the property
print (X.groupby('LandContour').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('LandContour').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Flatness of the property",size=13)
plt.tight_layout()

sns.boxplot('LandContour', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS LandContour",size=13)
plt.tight_layout()
## this is quite intersting to me. Banked houses seem to have lower prices overall. might keep this.

##6. Utilities: Type of utilities available [drop it]
print (X.groupby('Utilities').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('Utilities').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Type of utilities available",size=13)
plt.tight_layout()

sns.boxplot('Utilities', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS Utilities",size=13)
plt.tight_layout()

##7. LotConfig: Lot configuration [drop it]
print (X.groupby('LotConfig').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('LotConfig').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Lot Configuration",size=13)
plt.tight_layout()

sns.boxplot('LotConfig', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS LotConfig",size=13)
plt.tight_layout()

##8. LandSlope: Slope of the property [drop it]
print (X.groupby('LandSlope').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('LandSlope').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Slope of property",size=13)
plt.tight_layout()

sns.boxplot('LandSlope', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS LandSlope",size=13)
plt.tight_layout()

##9. Neighborhood: Physical locations within Ames city [keep]
plt.figure(figsize = (11, 4))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = X_train)
xt = plt.xticks(rotation=90)
plt.tight_layout()

## Condition1-2: Proximity to various conditions [keep it]
print (X.groupby('Condition1').size().sort_values(ascending=False))
print ('--------------')
print (X.groupby('Condition2').size().sort_values(ascending=False))

fig, ax = plt.subplots(2, 2, figsize = (11, 8))

X.groupby('Condition1').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("Condition 1 - proximity to various conditions",size=13)
plt.tight_layout()

X.groupby('Condition2').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1]) 
plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("Condition 2 - proximity to various conditions",size=13)
plt.tight_layout()

sns.boxplot('Condition1', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 400000))
ax[1][0].set_title("SalePrice VS Condition1",size=13)
plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)
plt.tight_layout()

sns.boxplot('Condition2', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 400000))
ax[1][1].set_title("SalePrice VS Condition2",size=13)
plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)
plt.tight_layout()

## BldgType: Type of dwelling + House Style: Style of dwelling [keep style of dwelling]
print (X.groupby('BldgType').size().sort_values(ascending=False))
print ('--------------')
print (X.groupby('HouseStyle').size().sort_values(ascending=False))

fig, ax = plt.subplots(2, 2, figsize = (11, 8))

X.groupby('BldgType').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("Type of dwelling",size=13)
plt.tight_layout()

X.groupby('HouseStyle').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1])
plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("Style of dwelling",size=13)
plt.tight_layout()

sns.boxplot('BldgType', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 400000))
ax[1][0].set_title("SalePrice VS BldgType",size=13)
plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)
plt.tight_layout()

sns.boxplot('HouseStyle', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 400000))
ax[1][1].set_title("SalePrice VS HouseStyle",size=13)
plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)
plt.tight_layout()

## RoofStyle [drop] / RoofMatl [drop]
print (X.groupby('RoofStyle').size().sort_values(ascending=False))
print ('--------------')
print (X.groupby('RoofMatl').size().sort_values(ascending=False))

fig, ax = plt.subplots(2, 2, figsize = (11, 8))

X.groupby('RoofStyle').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("Type of roof",size=13)
plt.tight_layout()

X.groupby('RoofMatl').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1])
plt.setp(ax[0][1].get_xticklabels(),rotation=90,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("Roof material",size=13)
plt.tight_layout()

sns.boxplot('RoofStyle', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 500000))
ax[1][0].set_title("SalePrice VS RoofStyle",size=13)
plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)
plt.tight_layout()

sns.boxplot('RoofMatl', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 500000))
ax[1][1].set_title("SalePrice VS RoofMatl",size=13)
plt.setp(ax[1][1].get_xticklabels(),rotation=90,size=12)
plt.tight_layout()

##Exterior1st / Exterior2nd [drop]
print (X.groupby('Exterior1st').size().sort_values(ascending=False))
print ('--------------')
print (X.groupby('Exterior2nd').size().sort_values(ascending=False))

fig, ax = plt.subplots(2, 2, figsize = (11, 8))

X.groupby('Exterior1st').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=45,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("Exterior1st - exterior covering on house",size=13)
plt.tight_layout()

X.groupby('Exterior2nd').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1]) 
plt.setp(ax[0][1].get_xticklabels(),rotation=45,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("Exterior2nd - exterior covering on house (if more than 1 material)",size=13)
plt.tight_layout()

sns.boxplot('Exterior1st', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 400000))
ax[1][0].set_title("SalePrice VS Exterior1st",size=13)
plt.setp(ax[1][0].get_xticklabels(),rotation=45,size=12)
plt.tight_layout()

sns.boxplot('Exterior2nd', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 400000))
ax[1][1].set_title("SalePrice VS Exterior2nd",size=13)
plt.setp(ax[1][1].get_xticklabels(),rotation=45,size=12)
plt.tight_layout()

##MasVnrType (Masonry veneer type) / MasVnrArea (Masonry veneer area) [drop]
print (X.groupby('MasVnrType').size().sort_values(ascending=False))
print ('-----------------------------')
print (X["MasVnrArea"].skew())

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('MasVnrType').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Masonry veneer type",size=13)
plt.tight_layout()

ax[1].scatter(range(X.shape[0]), X["MasVnrArea"].values,color='orange')
ax[1].set_title("Distribution of MasVnrArea", size=13)
ax[1].set_xlabel("Number of Occurences", size=12)
ax[1].set_ylabel("MasVnrArea, Square Feet", size=12)
plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[1].get_yticklabels(),size=12)
plt.tight_layout()

##ExterQual / ExterCond [drop]

##Foundation: Type of foundation [drop]
print (X.groupby('Foundation').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('Foundation').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Type of foundation",size=13)
plt.tight_layout()

sns.boxplot('Foundation', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 800000))
ax[1].set_title("SalePrice VS Foundation",size=13)
plt.tight_layout()

##BsmtQual (height of the basement) [keep] / BsmtCond (general condition of the basement) [drop]
print (X.groupby('BsmtQual').size())
print ('--------------')
print (X.groupby('BsmtCond').size())

fig, ax = plt.subplots(1,2, figsize = (11, 4))

sns.boxplot('BsmtQual', 'SalePrice', data = X_train, ax = ax[0]).set(ylim = (0, 500000))
ax[0].set_title("SalePrice VS BsmtQual",size=13)
plt.tight_layout()

sns.boxplot('BsmtCond', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS BsmtCond",size=13)
plt.tight_layout()

##BsmtExposure: Refers to walkout or garden level walls [drop]
print (X.groupby('BsmtExposure').size())

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('BsmtExposure').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("BsmtExposure - refers to walkout or garden level walls",size=13)
plt.tight_layout()

sns.boxplot('BsmtExposure', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 800000))
ax[1].set_title("SalePrice VS BsmtExposure",size=13)
plt.tight_layout()

##BsmtFinType1 [keep], BsmtFinType2 [drop] - Rating of basement finished area / Rating of basement finished area (if multiple types)
print (X.groupby('BsmtFinType1').size())
print ('--------------')
print (X.groupby('BsmtFinType2').size())

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

sns.boxplot('BsmtFinType1', 'SalePrice', data = X_train, ax = ax[0]).set(ylim = (0, 800000))
ax[0].set_title("SalePrice VS BsmtFinType1",size=13)
plt.tight_layout()

sns.boxplot('BsmtFinType2', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 800000))
ax[1].set_title("SalePrice VS BsmtFinType2",size=13)
plt.tight_layout()

##Type of heating / Heating Quality [drop]
print (X.groupby('Heating').size().sort_values(ascending=False))
print ('--------------')
print (X.groupby('HeatingQC').size())

fig, ax = plt.subplots(3, 2, figsize = (11, 12))

X.groupby('Heating').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("Heating - type of heating",size=13)
plt.tight_layout()

X.groupby('HeatingQC').size().plot(kind='bar', ax=ax[0][1]) 
plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("HeatingQC - heating quality",size=13)
plt.tight_layout()

sns.boxplot('Heating', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 500000))
ax[1][0].set_title("SalePrice VS type of heating",size=13)
plt.tight_layout()

sns.violinplot('HeatingQC', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 500000))
ax[1][1].set_title("SalePrice VS heating quality",size=13)
plt.tight_layout()

sns.stripplot("HeatingQC", "SalePrice",data = X_train, hue = 'CentralAir', jitter=True, split=True, ax = ax[2][0]).set(ylim = (0, 600000))
ax[2][0].set_title("Sale Price vs Heating Quality / Air Conditioning",size=13)
plt.tight_layout()

##CentralAir: central air conditioning [keep]
print (X.groupby('CentralAir').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('CentralAir').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("CentralAir - central air conditioning",size=13)
plt.tight_layout()

sns.boxplot('CentralAir', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS CentralAir",size=13)
plt.tight_layout()

##Electrical: Electrical system [drop]
print (X.groupby('Electrical').size().sort_values(ascending=False))

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('Electrical').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Electrical - type of electrical system",size=13)
plt.tight_layout()

sns.boxplot('Electrical', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS Electrical",size=13)
plt.tight_layout()

##1stFlrSF / 2ndFlrSF: First floor area / Second floor area [make new feature]

##BsmtFullBath / BsmtHalfBath / FullBath / HalfBath [make new feature]
print (X.groupby('BsmtFullBath').size())
print ('------------')
print (X.groupby('BsmtHalfBath').size())
print ('------------')
print (X.groupby('FullBath').size())
print ('------------')
print (X.groupby('HalfBath').size())

fig, ax = plt.subplots(2, 2, figsize = (11, 8))

X.groupby('BsmtFullBath').size().plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("BsmtFullBath - basement full bathrooms",size=13)
plt.tight_layout()

X.groupby('BsmtHalfBath').size().plot(kind='bar', ax=ax[0][1]) 
plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("BsmtHalfBath - basement half bathrooms",size=13)
plt.tight_layout()

X.groupby('FullBath').size().plot(kind='bar', ax=ax[1][0])
plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[1][0].get_yticklabels(),size=12)
ax[1][0].set_ylabel('Number of houses',size=12)
ax[1][0].set_title("FullBath - full bathrooms above grade",size=13)
plt.tight_layout()

X.groupby('HalfBath').size().plot(kind='bar', ax=ax[1][1]) 
plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[1][1].get_yticklabels(),size=12)
ax[1][1].set_ylabel('Number of houses',size=12)
ax[1][1].set_title("HalfBath - half bathrooms above grade",size=13)
plt.tight_layout()

##BedroomAbvGr: Bedrooms above grade (w/o basement bedrooms) [drop]
print (X.groupby('BedroomAbvGr').size())

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('BedroomAbvGr').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Bedroom - number of bedrooms above grade",size=13)
plt.tight_layout()

sns.boxplot('BedroomAbvGr', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS BedroomAbvGr",size=13)
plt.tight_layout()

##Kitchen [drop] / Kitchen Quality [keep]
print (X.groupby('KitchenAbvGr').size())
print ('--------------')
print (X.groupby('KitchenQual').size())

fig, ax = plt.subplots(2, 2, figsize = (11, 8))

X.groupby('KitchenAbvGr').size().plot(kind='bar', ax=ax[0][0])
plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][0].get_yticklabels(),size=12)
ax[0][0].set_ylabel('Number of houses',size=12)
ax[0][0].set_title("KitchenAbvGr - number of kitchen",size=13)
plt.tight_layout()

X.groupby('KitchenQual').size().plot(kind='bar', ax=ax[0][1]) 
plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0][1].get_yticklabels(),size=12)
ax[0][1].set_ylabel('Number of houses',size=12)
ax[0][1].set_title("KitchenQual - kitchen quality",size=13)
plt.tight_layout()

sns.boxplot('KitchenAbvGr', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 500000))
ax[1][0].set_title("SalePrice VS KitchenAbvGr",size=13)
plt.tight_layout()

sns.boxplot('KitchenQual', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 500000))
ax[1][1].set_title("SalePrice VS KitchenQual",size=13)
plt.tight_layout()

##TotRmsAbvGrd - Total Rooms above grade [keep]
fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('TotRmsAbvGrd').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("TotRmsAbvGrd - number of rooms above grade",size=13)
plt.tight_layout()

sns.boxplot('TotRmsAbvGrd', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS TotRmsAbvGrd",size=13)
plt.tight_layout()

##Functional - home functionality [drop]
print (X.groupby('Functional').size())
print ('--------------')

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('Functional').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Functional - home functionality",size=13)
plt.tight_layout()

sns.boxplot('Functional', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS Functional",size=13)
plt.tight_layout()

##GarageType - Garage location [keep]
print (X.groupby('GarageType').size())
print ('--------------')

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('GarageType').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("GarageType - garage location",size=13)
plt.tight_layout()

sns.boxplot('GarageType', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS GarageType",size=13)
plt.tight_layout()

##GarageYrBlt / GarageFinish - Year garage was built [make new feature] / Interior finish of the garage [keep]
fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('GarageFinish').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of garages',size=12)
ax[0].set_title("GarageFinish - interior finish of the garage",size=13)
plt.tight_layout()

sns.boxplot('GarageFinish', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS GarageFinish",size=13)
plt.tight_layout()

##GarageCars [keep] / GarageArea - Size of garage in cars / in square feet
print (X.groupby('GarageCars').size().sort_values(ascending=False))
print ('-----------------------------')
print (X["GarageArea"].skew())

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('GarageCars').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("Garage capacity in cars",size=13)
plt.tight_layout()

sns.boxplot('GarageCars', 'SalePrice', data = X_train, order = [2,1,3,0,4],ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS GarageCars",size=13)
plt.tight_layout()

##GarageQual / GarageCond [drop both]
print (X.groupby('GarageQual').size())
print ('--------------')
print (X.groupby('GarageCond').size())

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

sns.boxplot('GarageQual', 'SalePrice', data = X_train, ax = ax[0]).set(ylim = (0, 500000))
ax[0].set_title("SalePrice VS GarageQual",size=13)
plt.tight_layout()

sns.boxplot('GarageCond', 'SalePrice', data = X_train, ax = ax[1]).set(ylim = (0, 500000))
ax[1].set_title("SalePrice VS GarageCond",size=13)
plt.tight_layout()

##PavedDrive - Paved driveway [keep]
print (X.groupby('PavedDrive').size())
print ('--------------')

fig, ax = plt.subplots(1, 2, figsize = (11, 4))

X.groupby('PavedDrive').size().plot(kind='bar', ax=ax[0])
plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[0].get_yticklabels(),size=12)
ax[0].set_ylabel('Number of houses',size=12)
ax[0].set_title("PavedDrive - driveway",size=13)
plt.tight_layout()

sns.boxplot('PavedDrive', 'SalePrice', data = X_train, order = ['N','P','Y'],ax = ax[1]).set(ylim = (0, 400000))
ax[1].set_title("SalePrice VS PavedDrive",size=13)
plt.tight_layout()

##MoSold / YrSold - month and year sold
fig, ax = plt.subplots(5, 2, figsize = (11, 16))

sns.countplot(x = 'YrSold', data = X, ax=ax[0][0])
plt.tight_layout()

sns.countplot(x = 'MoSold', data = X, ax=ax[0][1])
plt.tight_layout()

sns.boxplot('YrSold', 'SalePrice', data = X_train, ax = ax[1][0]).set(ylim = (0, 400000))
plt.tight_layout()

sns.boxplot('MoSold', 'SalePrice', data = X_train, ax = ax[1][1]).set(ylim = (0, 400000))
plt.tight_layout()

X[X['YrSold']==2006].groupby('MoSold').size().plot(kind='bar', ax=ax[2][0])
plt.setp(ax[2][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[2][0].get_yticklabels(),size=12)
ax[2][0].set_ylabel('Number of houses',size=12)
ax[2][0].set_title("Sold in 2006",size=13)
ax[2][0].set_ylim(0, 80)
plt.tight_layout()

X[X['YrSold']==2007].groupby('MoSold').size().plot(kind='bar', ax=ax[2][1])
plt.setp(ax[2][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[2][1].get_yticklabels(),size=12)
ax[2][1].set_ylabel('Number of houses',size=12)
ax[2][1].set_title("Sold in 2007",size=13)
ax[2][1].set_ylim(0, 80)
plt.tight_layout()

X[X['YrSold']==2008].groupby('MoSold').size().plot(kind='bar', ax=ax[3][0])
plt.setp(ax[3][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[3][0].get_yticklabels(),size=12)
ax[3][0].set_ylabel('Number of houses',size=12)
ax[3][0].set_title("Sold in 2008",size=13)
ax[3][0].set_ylim(0, 80)
plt.tight_layout()

X[X['YrSold']==2009].groupby('MoSold').size().plot(kind='bar', ax=ax[3][1])
plt.setp(ax[3][1].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[3][1].get_yticklabels(),size=12)
ax[3][1].set_ylabel('Number of houses',size=12)
ax[3][1].set_title("Sold in 2009",size=13)
ax[3][1].set_ylim(0, 80)
plt.tight_layout()

X[X['YrSold']==2010].groupby('MoSold').size().plot(kind='bar', ax=ax[4][0])
plt.setp(ax[4][0].get_xticklabels(),rotation=0,size=12)
plt.setp(ax[4][0].get_yticklabels(),size=12)
ax[4][0].set_ylabel('Number of houses',size=12)
ax[4][0].set_title("Sold in 2010",size=13)
ax[4][0].set_ylim(0, 80)
ax[4][0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.tight_layout()
##interesting that summer months tend to sell more houses. but doesnt affect price. so drop.

##make new features
X['houseAge_fg'] = X['YrSold'] - X['YearBuilt']
X['totBaths_fg'] = X['FullBath'] + X['BsmtFullBath'] +\
    X['HalfBath']*0.5 + X['BsmtHalfBath']*0.5
X['totFlrSF_fg'] = X['1stFlrSF'] + X['2ndFlrSF']
X['totLot_fg'] = X['LotFrontage'] + X['LotArea']
X['garageAge_fg'] = X['YrSold'] - X['GarageYrBlt']
    
dropVars = ['SaleCondition', 'SaleType', 'MoSold', 'MiscVal', 'PoolArea',\
            'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'OpenPorchSF',\
            'WoodDeckSF', 'GarageCond', 'GarageQual', 'GarageArea',\
            'Fireplaces', 'Functional', 'KitchenAbvGr', 'BedroomAbvGr',\
            'Electrical', 'Heating', 'HeatingQC', 'BsmtFinType2',\
            'BsmtExposure', 'BsmtCond', 'Foundation', 'ExterCond',\
            'ExterQual', 'MasVnrArea', 'MasVnrType', 'Exterior1st', \
            'Exterior2nd', 'RoofMatl', 'RoofStyle', 'BldgType', \
            'LandSlope', 'LotConfig', 'Utilities', 'LotShape', \
            'Street', 'MSSubClass', 'MSZoning', 'YrSold', 'BsmtHalfBath',\
            '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'HalfBath',\
            'GarageYrBlt', 'YearBuilt']

X = X.drop(columns = dropVars)

#split data back to test and train data sets
X_train=X.iloc[:trainrow]
X_test=X.iloc[trainrow:]

X_train.insert(0, 'Id', trainids)
X_train['SalePrice'] = y_train
X_train.to_csv("train_cleaned.csv", index = False)


X_test.insert(0, 'Id', df_test['Id'].values)
X_test.to_csv('test_cleaned.csv', index = False)

#scaler=StandardScaler()
#scaler=scaler.fit(x_train)
#x_train_scaled=scaler.transform(x_train)
#x_test_scaled=scaler.transform(x_test)
