
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer as imp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score as cv
from sklearn.preprocessing import RobustScaler 
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from time import time
import xgboost


###EDA

#Load and check the source data
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

#This column is solely to ensure the data stays segregated during feature 
#engineering, scaling etc. It's probably unnecessary but included for peace
#of mind. 
x_train['Train'] = 1
test['Train'] = 0

#sanity check
print(x_train.shape)
print(test.shape)
print(y_train.shape)

#merge for eda and feature engineering
all_data = pd.concat([x_train, test])
print(all_data.shape)

#Missingness Summary
null_cols = all_data.isnull().sum()
null_cols.loc[null_cols > 0]

#Lot frontage imputation. A more granular approach would be preferable.
lot_frontage = np.array(all_data['LotFrontage'])
lot_frontage = lot_frontage.reshape(-1,1)
impute = imp(missing_values = 'NaN', strategy = 'mean', axis = 0)
impute = impute.fit(lot_frontage)
lot_frontage = (impute.transform(lot_frontage))
lot_frontage = np.round(lot_frontage)
all_data['LotFrontage'] = lot_frontage

#verifying data types
for c in all_data.columns:
    print(c, all_data[c].dtype)
    
#Missingness
cols_to_zero = ['BsmtFinSF1','MasVnrArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']

for col in cols_to_zero:
    mask_zero = all_data[col].isnull()    
    all_data.loc[mask_zero, col] = 0

mask_veneer = all_data['MasVnrType'].isnull()    
all_data.loc[mask_veneer, 'MasVnrType'] = 'None'

mask_garage = all_data['GarageYrBlt'].isnull()
all_data.loc[mask_garage, 'GarageYrBlt'] = all_data['YearBuilt'].min() - 1

mask_electrical = all_data['Electrical'].isnull()    
all_data.loc[mask_electrical, 'Electrical'] = 'Unknown'


#These are all missing exclusively from the test set. Values selected based on
#manual analysis in excel
mask_garage_type = all_data['GarageType'].isnull()
all_data.loc[mask_garage_type, 'GarageType'] = 'None'

mask_fence = all_data['Fence'].isnull()    
all_data.loc[mask_fence, 'Fence'] = 'None'

mask_misc = all_data['MiscFeature'].isnull()    
all_data.loc[mask_misc, 'MiscFeature'] = 'None'

mask_zoning = all_data['MSZoning'].isnull()    
all_data.loc[mask_zoning, 'MSZoning'] = 'RL'

mask_ext1 = all_data['Exterior1st'].isnull()    
all_data.loc[mask_ext1, 'Exterior1st'] = 'Wd Sdng'

mask_ext2 = all_data['Exterior2nd'].isnull()    
all_data.loc[mask_ext2, 'Exterior2nd'] = 'Wd Sdng'

mask_sale = all_data['SaleType'].isnull()    
all_data.loc[mask_sale, 'SaleType'] = 'WD'


#Ordinals
ord_cols = ['ExterQual', 'ExterCond','BsmtQual','BsmtCond','HeatingQC', 'KitchenQual', 
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


#At this point, all nulls should be adressed, and the data should be Nan free
nulls = all_data.isnull().sum()
nulls[nulls > 0]
all_data[all_data.isnull().any(axis=1)]

###Feature Engineering

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype('int64')
all_data['LotFrontage'] = all_data['LotFrontage'].astype('int64')
all_data['MasVnrArea'] = all_data['MasVnrArea'].astype('int64')


all_data['PorchSF'] = all_data['WoodDeckSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch'] + all_data['OpenPorchSF']

#Convert individual Porch/Deck Columns to dummies as the square footage has been
#aggregated
wood_mask = all_data['WoodDeckSF'] > 0
all_data.loc[wood_mask, 'WoodDeckSF'] = 1

enclosed_mask = all_data['EnclosedPorch'] > 0
all_data.loc[enclosed_mask, 'EnclosedPorch'] = 1

three_mask = all_data['3SsnPorch'] > 0
all_data.loc[three_mask, '3SsnPorch'] = 1

screen_mask = all_data['ScreenPorch'] > 0
all_data.loc[screen_mask, 'ScreenPorch'] = 1

open_mask = all_data['OpenPorchSF'] > 0
all_data.loc[open_mask, 'OpenPorchSF'] = 1

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['MSSubClass'] = all_data['MSSubClass'].astype('object') 

all_data['YrSold'] = all_data['YrSold'] - 2005

remod_mask = all_data['YearRemodAdd'] != all_data['YearBuilt']
all_data['Remodel'] = 0
all_data.loc[remod_mask, 'Remodel'] = 1
all_data.drop('YearRemodAdd', axis = 1, inplace = True)

#Dummies for remaining categoricals
for c in all_data.columns:
    if all_data[c].dtype == 'object':
        print(c, len(all_data[c].value_counts()))
        
one_hot_df = pd.get_dummies(all_data, drop_first=True, dummy_na=True)
one_hot_df.head()

###Scaling the remaining numeric variables

scale_variables = ['LotFrontage', 'LotArea', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'PorchSF', 'PoolArea', 'TotalSF']


mask_train = one_hot_df['Train'] == 1
mask_test = one_hot_df['Train'] == 0

#I experimented with Random forests on unscaled data. There was no significant
#improvement, and it seemed desirable to run all models off the same version 
#of the data, so this is only used as an intermediate step
x_train_no_scale = one_hot_df.loc[mask_train,]
x_train_no_scale.drop('Train', axis = 1, inplace = True)
x_train_no_scale.shape

x_test_no_scale = one_hot_df.loc[mask_test,]
x_test_no_scale.drop('Train', axis = 1, inplace = True)

#Scaling 
x_train_robust = x_train_no_scale.copy()
x_test_robust = x_test_no_scale.copy()
robust = RobustScaler()

x_test_robust[scale_variables] = robust.fit_transform(x_test_robust[scale_variables])
x_train_robust[scale_variables] = robust.fit_transform(x_train_robust[scale_variables])


#CSV data shared
x_train_robust.to_csv('x_train_robust.csv', index = False)
x_test_robust.to_csv('x_test_robust.csv', index = False)



###Modelling

#Function for reporting grid search results
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#function for returning feature importance sorted
def features(model, data):
    feature_importance = list(zip(data.columns, model.feature_importances_))
    dtype = [('feature', 'S10'), ('importance', 'float')]
    feature_importance = np.array(feature_importance, dtype=dtype)
    feature_sort = np.sort(feature_importance, order='importance')[::-1]
    return feature_sort



#Regression

reg1 = LR()
reg1.fit(x_train_robust, np.log1p(y_train))
reg1.score(x_train_robust, np.log1p(y_train))

#Cross Validation scoring
cvtest = cv(reg1, x_train_robust, np.log1p(y_train),scoring="neg_mean_squared_error", cv = 5)
np.expm1(-cvtest.mean())

#Feature importance
features(reg1, x_train_robust)

#Prediction and csv output
reg1p = np.round(np.expm1(reg1.predict(x_test_robust)))
pd.DataFrame({'Id': test_ID, 'SalePrice': reg1p}).to_csv('reg1p.csv', index = False)

#Random Forest

rf_search = RandomForestRegressor()

param_grid = {"max_depth": [3, 5, 10, None],
              "n_estimators": [800, 1000, 1200],
              "max_features": [1, 3, 10, 'sqrt', 'auto'],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False]}

# run grid search
grid_search = GridSearchCV(rf_search, param_grid=param_grid)
start = time()
grid_search.fit(x_train_robust, np.log1p(y_train))

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

#Below params are output of grid search
rfOpt = RandomForestRegressor(n_estimators=800, max_features='sqrt', bootstrap=False)
rfOpt.fit(x_train_robust, np.log1p(y_train))
rfOpt.score(x_train_robust, np.log1p(y_train))

#Cross Validation scoring
cvtest = cv(rfOpt, x_train_robust, np.log1p(y_train),scoring="neg_mean_squared_error", cv = 5)
np.expm1(-cvtest.mean())

#Feature importance
features(rfOpt, x_train_robust)

#Prediction and csv output
rfOptp = np.round(np.expm1(rfOpt.predict(x_test_robust)))
pd.DataFrame({'Id': test_ID, 'SalePrice': rfOptp}).to_csv('rfOptp.csv', index = False)


#GBM


param_grid_gbm = {"max_depth": [3, 5, 10, None],
              "n_estimators": [800, 1000, 1200],
              "max_features": [10, 'sqrt', 'auto'],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3],
              "learning_rate": [0.1, 0.2, 0.01]}

# run grid search
gb1 = GradientBoostingRegressor()
grid_search = GridSearchCV(gb1, param_grid=param_grid_gbm, n_jobs=1)
start = time()
grid_search.fit(x_train_robust, np.log1p(y_train))

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

#Parameters output from grid search
gbOpt = GradientBoostingRegressor()
gbOpt.set_params(learning_rate = 0.01, max_depth = 5, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 10, n_estimators = 1200)
gbOpt.fit(x_train_robust, np.log1p(y_train))
gbOpt.score(x_train_robust, np.log1p(y_train))

#Cross Validation scoring
cvtest = cv(gbOpt, x_train_robust, np.log1p(y_train),scoring="neg_mean_squared_error", cv = 5)
np.expm1(-cvtest.mean())

#Feature importance
features(gbOpt, x_train_robust)

#Prediction and csv output
gbOptp = np.round(np.expm1(gbOpt.predict(x_test_robust)))
pd.DataFrame({'Id': test_ID, 'SalePrice': gbOptp}).to_csv('gbOptp.csv', index = False)


#XGBoost

xr = xgboost.XGBRegressor()

params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4], 'n_estimators': [100, 500, 800, 1000, 1200]}

#Grid Search
grid_search = GridSearchCV(xr, param_grid=params, n_jobs=1)
start = time()
grid_search.fit(x_train_robust, np.log1p(y_train))

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

xgbOpt = xgboost.XGBRegressor()
xgbOpt.set_params(colsample_bytree=1.0, gamma=0.3, max_depth=3, min_child_weight=4, n_estimators=800, subsample=0.6)
xgbOpt.fit(x_train_robust, np.log1p(y_train))
xgbOpt.score(x_train_robust, np.log1p(y_train))

#Cross Validation scoring
cvtest = cv(xgbOpt, x_train_robust, np.log1p(y_train),scoring="neg_mean_squared_error", cv = 5)
np.expm1(-cvtest.mean())

#Feature importance
features(xgbOpt, x_train_robust)

#Prediction and csv output
xgbOptp = np.round(np.expm1(xgbOpt.predict(x_test_robust)))
pd.DataFrame({'Id': test_ID, 'SalePrice': xgbOptp}).to_csv('xgbOptp.csv', index = False)


#Hasty Lasso and ridge
lasso = linear_model.Lasso(alpha=0.1)
ridge = linear_model.Ridge(alpha=0.1)

lasso.fit(x_train_robust, np.log1p(y_train))
ridge.fit(x_train_robust, np.log1p(y_train))
lasso.score(x_train_robust, np.log1p(y_train))
ridge.score(x_train_robust, np.log1p(y_train))
cv(lasso, x_train_robust, np.log1p(y_train), cv = 10)
cv(ridge, x_train_robust, np.log1p(y_train), cv = 10)
lassop = np.round(np.expm1(lasso.predict(x_test_robust)))
ridgep = np.round(np.expm1(ridge.predict(x_test_robust)))
pd.DataFrame({'Id': test_ID, 'SalePrice': lassop}).to_csv('lassop.csv', index = False)
pd.DataFrame({'Id': test_ID, 'SalePrice': ridgep}).to_csv('ridgep.csv', index = False)


#Attempt at stacking. Produced promising cv score and garbage kaggle score.
#Figure out what I did wrong, win a cookie
reg1xp = np.round(np.expm1(reg1.predict(x_train_robust)))
rfOptxp = np.round(np.expm1(rfOpt.predict(x_train_robust)))
gbOptxp = np.round(np.expm1(gbOpt.predict(x_train_robust)))
xgbOptxp = np.round(np.expm1(xgbOpt.predict(x_train_robust)))
train_predicts = pd.DataFrame({'reg1': reg1xp, 'rfOpt': rfOptxp, 'gbOpt': gbOptxp, 'xgbOpt': xgbOptxp})
test_predicts = pd.DataFrame({'reg1': reg1p, 'rfOpt': rfOptp, 'gbOpt': gbOptp, 'xgbOpt': xgbOptp})
test_predicts
train_predicts

stack_reg = LR()
stack_reg.fit(train_predicts, np.log1p(y_train))
cv(stack_reg, train_predicts, np.log1p(y_train), cv = 10)
stack_regp = np.round(np.expm1(stack_reg.predict(test_predicts)))
pd.DataFrame({'Id': test_ID, 'SalePrice': stack_regp}).to_csv('stack_regp.csv', index = False)
stack_regp





#Averaged/Weighted Average models. haphazard, but refined in response to kaggle scores

AverageTest = (rfOptp + reg1p * 2) / 3
AverageTest = np.round(AverageTest)
pd.DataFrame({'Id': test_ID, 'SalePrice': AverageTest}).to_csv('AverageTest.csv', index = False)


avgp = np.round((xgbOptp + gbOptp + rfOptp + reg1p) / 4)
wavgp = np.round((xgbOptp + 2 * gbOptp + rfOptp + 2 * reg1p) / 6)
avgp
wavgp
pd.DataFrame({'Id': test_ID, 'SalePrice': avgp}).to_csv('avgp.csv', index = False)
pd.DataFrame({'Id': test_ID, 'SalePrice': wavgp}).to_csv('wavgp.csv', index = False)




wavgp3 = np.round((xgbOptp + 3 * gbOptp + rfOptp + 3 * reg1p) / 8)
pd.DataFrame({'Id': test_ID, 'SalePrice': wavgp3}).to_csv('wavgp3.csv', index = False)

wavgp4 = np.round((xgbOptp + 4 * gbOptp + rfOptp + 4 * reg1p) / 10)
pd.DataFrame({'Id': test_ID, 'SalePrice': wavgp4}).to_csv('wavgp4.csv', index = False)
gbreg = np.round((gbOptp + reg1p) / 2)
pd.DataFrame({'Id': test_ID, 'SalePrice': gbreg}).to_csv('gbreg.csv', index = False)
gbregx = np.round((xgbOptp + 4 * gbOptp + 4 * reg1p) / 9)
pd.DataFrame({'Id': test_ID, 'SalePrice': gbregx}).to_csv('gbregx.csv', index = False)

gb2regx = np.round((xgbOptp + 6 * gbOptp + 3 * reg1p) / 10)
pd.DataFrame({'Id': test_ID, 'SalePrice': gb2regx}).to_csv('gb2regx.csv', index = False)
