# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:50:36 2018

@author: ssurya200
"""
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
#Loading the application data
app_zf_train = zipfile.ZipFile('Data/application_train.csv.zip') 
app_df_train = pd.read_csv(app_zf_train.open('application_train.csv'))

#X_train = app_df_train.iloc[:,:].values





app_zf_test = zipfile.ZipFile('Data/application_test.csv.zip') 
app_df_test = pd.read_csv(app_zf_test.open('application_test.csv'))


# loading the bureau data

bur_zf = zipfile.ZipFile('Data/bureau.csv.zip') 
bur_df = pd.read_csv(bur_zf.open('bureau.csv'))

# Loading bureau balance

bur_bal_zf = zipfile.ZipFile('Data/bureau_balance.csv.zip') 
bur_bal_df = pd.read_csv(bur_bal_zf.open('bureau_balance.csv'))


# Loading credit car balance

cc_bal_zf = zipfile.ZipFile('Data/credit_card_balance.csv.zip') 
cc_bal_df = pd.read_csv(cc_bal_zf.open('credit_card_balance.csv'))


# loading installment payments

pay_zf = zipfile.ZipFile('Data/installments_payments.csv.zip') 
pay_df = pd.read_csv(pay_zf.open('installments_payments.csv'))
pay_df = pay_df.loc[:, pay_df.columns != 'SK_ID_PREV']

# Loading cash balance

cash_bal_zf = zipfile.ZipFile('Data/POS_CASH_balance.csv.zip') 
cash_bal_df = pd.read_csv(cash_bal_zf.open('POS_CASH_balance.csv'))


#loading Previous application data

prev_app_zf = zipfile.ZipFile('Data/previous_application.csv.zip') 
prev_app_df = pd.read_csv(prev_app_zf.open('previous_application.csv'))
prev_app_df = prev_app_df.loc[:, prev_app_df.columns != 'SK_ID_PREV']




#Removing columns based on RF feature importance
rf_bur = ['SK_ID_CURR','AMT_ANNUITY','SK_BUREAU_ID']
rf_bb = ['SK_BUREAU_ID','MONTHS_BALANCE']
rf_cc = ['SK_ID_CURR','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_POS_CURRENT','AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT','AMT_TOTAL_RECEIVABLE','CNT_DRAWINGS_POS_CURRENT','CNT_INSTALMENT_MATURE_CUM','MONTHS_BALANCE']
rf_pre = ['SK_ID_CURR','AMT_ANNUITY','AMT_CREDIT','DAYS_DECISION','NAME_TYPE_SUITE','NAME_CLIENT_TYPE','SELLERPLACE_AREA','PRODUCT_COMBINATION']
rf_cash = ['SK_ID_CURR','MONTHS_BALANCE']

#rf_bur_df = bur_df[rf_bur]
#rf_bur_bal_df = bur_bal_df[rf_bb]
rf_cc_bal_df = cc_bal_df[rf_cc] 
rf_prev_app_df = prev_app_df[rf_pre]
rf_cash_bal_df = cash_bal_df[rf_cash]

# train data merge
com_bur_df = pd.merge(bur_df, bur_bal_df, how = 'left', on = 'SK_ID_BUREAU')
app_df_train_slice = app_df_train.iloc[0:100,:]
X_df_train = pd.merge(app_df_train, rf_cc_bal_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(X_df_train, rf_prev_app_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(X_df_train, pos_cash_bal_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(X_df_train, inst_df, how = 'left', on = 'SK_ID_CURR')
#X_df_train = pd.merge(X_df_train, com_bur_df, how = 'left', on = 'SK_ID_CURR')



# data cleansing
nans = X_df_train.isnull().sum()
nans_ar = nans.reshape(X_df_train.shape[1],1)
nan_tresh = 40
nan_perc = []
for i in range(len(X_df_train.columns)):
    nan_perc.append(np.round((nans[i].astype(float)/len(X_df_train)*100), decimals = 2))
    
    
nan_perc = np.asarray(nan_perc).reshape(X_df_train.shape[1],1)
nans_ar = np.append(nans_ar, nan_perc, axis = 1)
nans_ar = np.append(nans_ar, (nans_ar[:,1] > nan_tresh).reshape(X_df_train.shape[1],1) , axis = 1)
thres_col = np.asarray(np.where(nans_ar[:,2] == 1)).T
to_remove_col = []
for i in range(len(thres_col)):
    to_remove_col.append(X_df_train.columns.values.tolist()[thres_col[i,0]])



X_train = []
X_test = []
y_train = []


# extracting the feature after removing the columns with missing values  more than threshold
X_df_train = X_df_train.drop(to_remove_col, axis = 1)

# Auto filling nans with the value forward
X_df_train = X_df_train.fillna(method = 'ffill')
X_df_train = X_df_train.fillna(method = 'backfill')
#X_df_train = X_df_train.fillna(X_df_train.mean(), inplace=True)


# label encoding and onehot encoding

X_df_train = pd.get_dummies(X_df_train, drop_first = True)


X_df_SK = X_df_train.loc[:, X_df_train.columns != 'SK_ID_CURR']
X = np.array(X_df_SK.loc[:, X_df_SK.columns != 'TARGET'])

y = X_df_train['TARGET']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(X_train)


X_train = np.array(training_set_scaled)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier(random_state = 3)
parameters = {'max_depth' : [80 ,100],
              'max_features' : ['auto', 'sqrt'],
              'bootstrap' : [True], 
              'min_samples_leaf': [4, 5],
              'min_samples_split': [10, 12],
              'n_estimators' : [10, 50]
              }

parameter = {'max_depth' : [80]}
grid_search = GridSearchCV(estimator = rf_model,
                           param_grid = parameter,
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
estimators = ["auto","sqrt"]
for est_size in estimators :
    model = RandomForestClassifier(n_estimators = 50, 
                                  oob_score = True, 
                                  random_state =50, 
                                  max_features = "sqrt", 
                                  min_samples_leaf = 10)
    model.fit(X_train,y_train)
    y_pred_ra = model.predict(X_test)
    prob_y_4 = model.predict_proba(X_test)
    prob_y_4 = [p[1] for p in prob_y_4]
    print( "roc: ",roc_auc_score(y_test, prob_y_4) )
    print( "acc: ",accuracy_score(y_test, y_pred_ra) )
    #print( model.oob_score_, leaf_size )
'''
model = RandomForestClassifier(max_depth = 80, random_state = 3)
model.fit(X_train,y_train)


y_pred_ra = model.predict(X_test)
y_pred_ra = (y_pred_ra >0.5)
result = y_pred_ra.astype(int)

# AUC score

from sklearn.metrics import roc_auc_score
prob_y_4 = model.predict_proba(X_test)
prob_y_4 = [p[1] for p in prob_y_4]
print( roc_auc_score(y_test, prob_y_4) )

from sklearn.metrics import accuracy_score
print( accuracy_score(y_test, y_pred_ra) )
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_ra)

## save the model to disk
import pickle
filename = 'RF_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
model = pickle.load(open(filename, 'rb'))
result = model.score(X_test, y_test)
print(result)

'''
X_df_SK = X_df_train
# feature importance
plt.plot(0.1,1)
features = X_df_SK.columns.values[X_df_SK.columns != 'TARGET']
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), features[indices], color='b', align='center')
plt.yticks(range(len(indices)), importances[indices])
plt.xlabel('Relative Importance')

# select the important feature

important_feature = []
for i in range(len(indices)):
    if (importances[indices[-(i+1)]] >= 0.001):
        important_feature.append(features[indices[-(i+1)]])
#important_feature.append('TARGET')
X_df_rf = X_df_train_mod[important_feature]

