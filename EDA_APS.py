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

x_train['Train'] = 1
test['Train'] = 0

print(x_train.shape)
print(test.shape)
print(y_train.shape)


all_data = pd.concat([x_train, test])
print(all_data.shape)

null_cols = all_data.isnull().sum()
null_cols.loc[null_cols > 0]

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
    
    
mask_veneer = all_data['MasVnrType'].isnull()    
all_data.loc[mask_veneer, 'MasVnrType'] = 'None'

mask_veneer_area = all_data['MasVnrArea'].isnull()    
all_data.loc[mask_veneer_area, 'MasVnrArea'] = 0

mask_garage = all_data['GarageYrBlt'].isnull()
all_data.loc[mask_garage, 'GarageYrBlt'] = all_data['YearBuilt'].min() - 1

mask_electrical = all_data['Electrical'].isnull()    
all_data.loc[mask_electrical, 'Electrical'] = 'Unknown'

ord_cols = ['OverallQual','OverallCond','ExterQual', 'ExterCond','BsmtQual','BsmtCond','HeatingQC', 'KitchenQual', 
           'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1, 'NaN':0}

for col in ord_cols:
    all_data[col] = all_data[col].map(lambda x: ord_dic.get(x, 0))
    

paved_dic = {'Y': 2, 'P': 1, 'N': 0}
all_data['PavedDrive'] = all_data['PavedDrive'].map(lambda x: paved_dic.get(x, 0))

garage_dic = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NaN': 0}
all_data['GarageFinish'] = all_data['GarageFinish'].map(lambda x: garage_dic.get(x, 0))

func_dic = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
all_data['Functional'] = all_data['Functional'].map(lambda x: func_dic.get(x, 0))

bsmt_dic = {'NaN': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(lambda x: bsmt_dic.get(x, 0))
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(lambda x: bsmt_dic.get(x, 0))

expo_dic = {'NaN': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
all_data['BsmtExposure'] = all_data['BsmtExposure'].map(lambda x: expo_dic.get(x, 0))

slope_dic = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
all_data['LandSlope'] = all_data['LandSlope'].map(lambda x: slope_dic.get(x, 0))

util_dic = {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}
all_data['Utilities'] = all_data['Utilities'].map(lambda x: util_dic.get(x, 0))

lot_dic = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
all_data['LotShape'] = all_data['LotShape'].map(lambda x: lot_dic.get(x, 0))

alley_dic = {'NaN': 0, 'Grvl': 1, 'Pave': 2}
all_data['Alley'] = all_data['Alley'].map(lambda x: alley_dic.get(x, 0))

street_dic = {'Grvl': 0, 'Pave': 1}
all_data['Street'] = all_data['Street'].map(lambda x: street_dic.get(x, 0))

bool_dic = {'N': 0, 'Y': 1}
all_data['CentralAir'] = all_data['CentralAir'].map(lambda x: bool_dic.get(x, 0))

mask_garage_type = all_data['GarageType'].isnull()
all_data.loc[mask_garage_type, 'GarageType'] = 'None'

mask_fence = all_data['Fence'].isnull()    
all_data.loc[mask_fence, 'Fence'] = 'None'

mask_misc = all_data['MiscFeature'].isnull()    
all_data.loc[mask_misc, 'MiscFeature'] = 'None'


all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype('int64')
all_data['LotFrontage'] = all_data['LotFrontage'].astype('int64')
all_data['MasVnrArea'] = all_data['MasVnrArea'].astype('int64')
