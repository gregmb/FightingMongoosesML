# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd


path = 'C:/GitRepos/HousePrices/'
pd.set_option('display.max_columns',300)
train = pd.read_csv(path + 'train.csv')
print(train.head())
test = pd.read_csv(path + 'test.csv')
print(test.head())

#save the Id column
train_ID = train['Id']
test_ID = test['Id']

#drop id from dfs
train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

#Seperate the target variables
y_train = train['SalePrice']
x_train = train.drop('SalePrice', axis = 1)
y_test = test['SalePrice']
x_test = test.drop('SalePrice', axis = 1)

x_train['Train'] = 1
x_test['Train'] = 0

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

all_data = pd.concat([x_train, x_test])
print(all_data.shape)

all_data.isnull().sum()

from sklearn.preprocessing import Imputer as imp

lot_frontage = np.array(all_data['LotFrontage'])
lot_frontage = lot_frontage.reshape(-1,1)

impute = imp(missing_values = 'NaN', strategy = 'mean', axis = 0)
impute = impute.fit(lot_frontage)
lot_frontage = (impute.transform(lot_frontage))
lot_frontage = np.round(lot_frontage)
all_data['LotFrontage'] = lot_frontage

for c in all_data.columns:
    print(c, all_data[c].dtype)

print(test.columns)
