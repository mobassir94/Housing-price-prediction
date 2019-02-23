# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:04:02 2018

@author: MOBASSIR
"""


# Importing the libraries
import numpy as np
import pandas as pd
 

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score


 
 
#Importing the dataset
train = pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')

"""
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']


missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

missing = missing[missing < 690]

train.head()
train.describe()
"""


# target 
y_train = train.SalePrice

# drops the Id and SalePrice columns
X_train = train.drop(['Id','SalePrice'], axis= 1)

# drops Id column from test
X_test = test.drop(['Id'], axis= 1)


# code to encode object data(text data) using one-hot encoding(commonly used)
training = pd.get_dummies(X_train,drop_first = True)
testing = pd.get_dummies(X_test,drop_first = True)

# align command make sure that the columns in both the datasets are in same order
final_train, final_test = training.align(testing,join='inner',axis=1)
# to check the increased number of columns
final_train.shape

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(final_train) 


"""

# RandomForestRegressor:
for n in range(10,200,10):
    # define pipeline
    pipeline = make_pipeline(Imputer(), RandomForestRegressor(max_leaf_nodes=n,random_state=1))
    # cross validation score
    scores = cross_val_score(pipeline, final_train, y_train, scoring= 'neg_mean_absolute_error')
    print(n,scores)
"""


# XGBRegressor:
# define pipeline
pipeline = make_pipeline(Imputer(missing_values='NaN', strategy='most_frequent', axis=0), XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.01, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                            nthread = -1, random_state=1))




"""
# cross validation score
scores = cross_val_score(pipeline,final_train, y_train, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))"""
#Validation function


# GradientBoostingRegressor:(just another model)
# define pipeline
my_pipeline = make_pipeline(Imputer(missing_values='NaN', strategy='most_frequent', axis=0), GradientBoostingRegressor())
# cross validation score
score = cross_val_score(my_pipeline,final_train, y_train, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * score.mean()))





# fit and make predictions
pipeline.fit(final_train,y_train)
predictions= pipeline.predict(final_test)

print(predictions)



"""
# finding Pvalue from statsmodels
 
import statsmodels.formula.api as sm
 
regressor_OLS = sm.OLS(endog=final_test,exog = final_train).fit()
 
regressor_OLS.summary()

"""



my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('mobassir1.csv', index=False)