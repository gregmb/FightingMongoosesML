# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score as cv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression as LR
import xgboost


x_train_robust = pd.read_csv('x_train_robust.csv')
x_test_robust = pd.read_csv('x_test_robust.csv')


reg1 = LR()
reg1.fit(x_train_robust, np.log1p(y_train))
reg1.score(x_train_robust, np.log1p(y_train))
cv(reg1, x_train_robust, np.log1p(y_train), cv = 10)
reg1p = np.round(np.expm1(reg1.predict(x_test_robust)))





rfOpt = RandomForestRegressor(n_estimators=800, max_features='sqrt', bootstrap=False)
rfOpt.fit(x_train_robust, np.log1p(y_train))
rfOpt.score(x_train_robust, np.log1p(y_train))
cv(rfOpt, x_train_robust, np.log1p(y_train), cv = 10)
rfOptp = np.round(np.expm1(rfOpt.predict(x_test_robust)))
#pd.DataFrame({'Id': test_ID, 'SalePrice': rfOptp}).to_csv('rfOptp.csv', index = False)


AverageTest = (rfOptp + reg1p * 2) / 3
AverageTest = np.round(AverageTest)
#pd.DataFrame({'Id': test_ID, 'SalePrice': AverageTest}).to_csv('AverageTest.csv', index = False)




gbOpt = GradientBoostingRegressor()
gbOpt.set_params(learning_rate = 0.01, max_depth = 5, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 10, n_estimators = 1200)
gbOpt.fit(x_train_robust, np.log1p(y_train))
gbOpt.score(x_train_robust, np.log1p(y_train))
cv(gbOpt, x_train_robust, np.log1p(y_train), cv = 10)
gbOptp = np.round(np.expm1(gbOpt.predict(x_test_robust)))
#pd.DataFrame({'Id': test_ID, 'SalePrice': gbOptp}).to_csv('gbOptp.csv', index = False)




xgb_params_result = {'colsample_bytree': 1.0, 'gamma': 0.3, 'max_depth': 3, 'min_child_weight': 4, 'n_estimators': 800, 'subsample': 0.6}

xgbOpt = xgboost.XGBRegressor()
xgbOpt.set_params(colsample_bytree=1.0, gamma=0.3, max_depth=3, min_child_weight=4, n_estimators=800, subsample=0.6)
xgbOpt.fit(x_train_robust, np.log1p(y_train))
xgbOpt.score(x_train_robust, np.log1p(y_train))
cv(xgbOpt, x_train_robust, np.log1p(y_train), cv = 10)
xgbOptp = np.round(np.expm1(xgbOpt.predict(x_test_robust)))
#pd.DataFrame({'Id': test_ID, 'SalePrice': xgbOptp}).to_csv('xgbOptp.csv', index = False)

avgp = np.round((xgbOptp + gbOptp + rfOptp + reg1p) / 4)
wavgp = np.round((xgbOptp + 2 * gbOptp + rfOptp + 2 * reg1p) / 6)
#pd.DataFrame({'Id': test_ID, 'SalePrice': avgp}).to_csv('avgp.csv', index = False)
#pd.DataFrame({'Id': test_ID, 'SalePrice': wavgp}).to_csv('wavgp.csv', index = False)


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
#pd.DataFrame({'Id': test_ID, 'SalePrice': stack_regp}).to_csv('stack_regp.csv', index = False)

wavgp3 = np.round((xgbOptp + 3 * gbOptp + rfOptp + 3 * reg1p) / 8)
#pd.DataFrame({'Id': test_ID, 'SalePrice': wavgp3}).to_csv('wavgp3.csv', index = False)

wavgp4 = np.round((xgbOptp + 4 * gbOptp + rfOptp + 4 * reg1p) / 10)
#pd.DataFrame({'Id': test_ID, 'SalePrice': wavgp4}).to_csv('wavgp4.csv', index = False)
gbreg = np.round((gbOptp + reg1p) / 2)
#pd.DataFrame({'Id': test_ID, 'SalePrice': gbreg}).to_csv('gbreg.csv', index = False)
gbregx = np.round((xgbOptp + 4 * gbOptp + 4 * reg1p) / 9)
#pd.DataFrame({'Id': test_ID, 'SalePrice': gbregx}).to_csv('gbregx.csv', index = False)

gb2regx = np.round((xgbOptp + 6 * gbOptp + 3 * reg1p) / 10)
#pd.DataFrame({'Id': test_ID, 'SalePrice': gb2regx}).to_csv('gb2regx.csv', index = False)


