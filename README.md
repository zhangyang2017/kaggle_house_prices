# Repo for the Kaggle "Housing Prices Competition"

Table of contents
- [Overview](#1)
- [Workflow](#2)
- [One more thought](#3)
- [Resources helped me learn](#4)


## Overview <a id="1"></a>
This was my first Kaggle competition, and also my very first ML project. I did not have the knowledge of the tools I need, nor did I know how to even approach the problem, but I wanted to jump into the data right away. My goal here, for this project, was to learn how to use python tools to build ML models that can solve regression problems like this: predict house prices based on a bunch of collected housing parameters.

In order to achieve my goal, the quickest way, I thought, was to study the public notebooks that people shared on Kaggle. I picked out three notebooks (which I listed towards the end) and read every line of codes. From doing that, I learned how to do exploratory data analysis (EDA) using [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/) and [seaborn](https://seaborn.pydata.org/), how to do feature selection and feature engineering, and how to build traditional ML models using [scikit-learn](https://scikit-learn.org/stable/) and a more robust model using [xgboost](https://xgboost.readthedocs.io/en/latest/).

After trying out a few different models, `XGBoost` gave me the best outcome. I eventually [scored at top 9%](https://www.kaggle.com/zhangyang2020/competitions). This was no where near perfect, but I decided it was good enough for me at the moment and it was time to move on to the next project. I do plan to come back to this later on after I gain more experience on ML and see if I can make my models any better :wink:.

Project duration: 2020/10/16 - 2020/10/20

## Workflow <a id="2"></a>
I ended up with a workflow like this:
  - EDA
    > This took quite some time, because I inspected every single variable and made a looot of plots
    - missing data imputation `pandas` `numpy`
    - normality and skewness `seaborn` `matplotlib`
    - correlation and heatmaps `scipy`
  - feature selection (drop redundant features and feature engineering) based on EDA
  - feature scaling / normalization / transformation `StandardScaler` `RobustScaler`
  - model selection and model training
    - `LinearRegression` `XGBRegressor`
  - model validation
  - prediction

## One more thought <a id="3"></a>
I wonder :thinking: which way is better:
 - 1. going through each variable, studying their distributions and relationships with the target variable so that I can make the 'best' decision on feature selection;
 - 2. or, giving my model enough flexibility to pick up information on its own?
 
 Does either of them lead to overfitting the model? :thinking:
  
## Resources helped me learn <a id="4"></a>

1. [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) by Pedro Marcelino
2. [Visualization + feature generation [House Prices]](https://www.kaggle.com/drsergio/visualization-feature-generation-house-prices) by Sergii Lutsanych
3. [Top 5% on Leaderboard](https://www.kaggle.com/vishalvanpariya/top-5-on-leaderboard) by Vishal Vanpariya

