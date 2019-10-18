# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:00:13 2018

@author: ssurya200
"""

import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score
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


# Loading cash balance

cash_bal_zf = zipfile.ZipFile('Data/POS_CASH_balance.csv.zip') 
cash_bal_df = pd.read_csv(cash_bal_zf.open('POS_CASH_balance.csv'))


#loading Previous application data

prev_app_zf = zipfile.ZipFile('Data/previous_application.csv.zip') 
prev_app_df = pd.read_csv(prev_app_zf.open('previous_application.csv'))


# data_type['NAME_TYPE_SUITE'].value_counts()
#np.where((bur_df.loc[bur_df['SK_ID_CURR'] == a[2,0]),1,np.nan) -- query
#a = pd.merge(bur_df, bur_bal_df, how = 'left', on = 'SK_ID_BUREAU') - merging two pandas
#b = bur_bal_df.loc[bur_bal_df['SK_ID_BUREAU'] == bur_df.iloc[5020,1]] -- to find the location
#a[a.SK_ID_BUREAU == 5720494]  -  to get entire row based on a column value
# bur_bal_df.nunique() --- to find the uniques values in each column


# merging two burea information
# finding the uniques bureau inforamtoin by extractin only -1s


# Cleansing large set
# installment payment
pay_df_mod = pay_df.dropna(axis = 0, how = 'any')
# adding paid on time column. 1 means on time, 0 means missed oayment
pay_df_mod.loc[:,'LATE_PAYMENT'] = np.where(pay_df_mod.DAYS_INSTALMENT >= pay_df_mod.DAYS_ENTRY_PAYMENT, 0, 1)
#missed payment is 1 
pay_df_mod.loc[:,'MISSED_PAYMENT'] = np.where(pay_df_mod.AMT_PAYMENT == pay_df_mod.AMT_INSTALMENT, 0, (pay_df_mod.AMT_INSTALMENT-pay_df_mod.AMT_PAYMENT))

# groupby df.groupby(['Code', 'Country', 'Item_Code', 'Item', 'Ele_Code', 'Unit']).agg({'Y1961': np.sum, 'Y1962': np.sum, 'Y1963': np.sum})

inst_df1 = pay_df_mod.groupby('SK_ID_CURR')[['LATE_PAYMENT','MISSED_PAYMENT']].sum().reset_index()



# cash balance


#recent information
idx_recent_poscash = cash_bal_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
recent_poscash_df = pd.DataFrame(cash_bal_df.loc[idx_recent_poscash].values)
recent_poscash_df.columns = list(cash_bal_df.columns)
recent_poscash_df = recent_poscash_df.add_prefix('recent_')
recent_poscash_df.rename(columns={'recent_SK_ID_CURR':'SK_ID_CURR'}, inplace=True)
#recent_poscash_df = recent_poscash_df.dropna(axis =0)
recent_poscash_df = recent_poscash_df.loc[:, recent_poscash_df.columns != 'recent_SK_ID_PREV']



pos_cash_bal_df1 = cash_bal_df.groupby('SK_ID_CURR')[['SK_DPD','SK_DPD_DEF']].sum()

pos_cash_bal_df2 = cash_bal_df.groupby('SK_ID_CURR')[['SK_ID_PREV']].nunique()

pos_cash_bal_df = pd.concat([pos_cash_bal_df1,pos_cash_bal_df2], axis = 1)

pos_cash_bal_df['SK_ID_CURR'] = pos_cash_bal_df.index

pos_cash_bal_df = pos_cash_bal_df.rename(index=str, columns = {'SK_ID_PREV' : 'CB_SK_ID_PREV_CNT',
                                                               'SK_DPD' : 'SK_DPD_poscash',
                                                               'SK_DPD_DEF' : 'SK_DPD_DEF_poscash'})

pos_cash_bal_df = pd.merge(pos_cash_bal_df, recent_poscash_df, how = 'left', on = 'SK_ID_CURR')

pos_cash_bal_df = pos_cash_bal_df.dropna(axis =0)


# Credit Card Balance


mod_cc_bal_df = pd.concat([cc_bal_df.groupby('SK_ID_CURR')[['SK_DPD','SK_DPD_DEF']].sum(), 
                        (cc_bal_df.groupby('SK_ID_CURR')[['MONTHS_BALANCE']].min()),
                        (cc_bal_df.sort_values('MONTHS_BALANCE', ascending = False).groupby('SK_ID_CURR')['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL'].first()),
                        (cc_bal_df.groupby('SK_ID_CURR')[['CNT_INSTALMENT_MATURE_CUM']].max())],axis = 1)



mod_cc_bal_df['SK_ID_CURR'] = mod_cc_bal_df.index



# previous applicaiton

# one hot encoding of categorical data and summing them
cat_prev_app_df = prev_app_df.select_dtypes(include = ['object'])
cat_prev_app_df['SK_ID_CURR'] = prev_app_df.SK_ID_CURR
#cat_prev_app_df = cat_prev_app_df.dropna(thresh = len(cat_prev_app_df) - len(cat_prev_app_df)*.2, axis = 1)
cat_prev_app_df = pd.get_dummies(cat_prev_app_df)
col_list = list(cat_prev_app_df.columns)
col_list.remove('SK_ID_CURR')
cat_prev_app_df = cat_prev_app_df.groupby('SK_ID_CURR')[col_list].sum()


# latest app information from prev app

idx_recent_app = prev_app_df.groupby('SK_ID_CURR')['DAYS_DECISION'].idxmax()
recent_app_df = pd.DataFrame(prev_app_df.loc[idx_recent_app].values)
recent_app_df.columns = list(prev_app_df.columns)
recent_app_df = recent_app_df.add_prefix('recent_')
recent_app_df.rename(columns={'recent_SK_ID_CURR':'SK_ID_CURR'}, inplace=True)
recent_app_df = recent_app_df.dropna(axis =1)
recent_app_df = recent_app_df.loc[:, recent_app_df.columns != 'recent_SK_ID_PREV']


#summing the amount of prev application and count of prev app

mod_prev_app_df = pd.concat([prev_app_df.groupby('SK_ID_CURR')[['AMT_APPLICATION']].sum(),
                             prev_app_df.groupby('SK_ID_CURR')[['AMT_CREDIT']].sum(),
                             prev_app_df.groupby('SK_ID_CURR')[['AMT_DOWN_PAYMENT']].sum(),], axis = 1)
mod_prev_app_df['AMT_ANNUITY_prev'] = prev_app_df.groupby('SK_ID_CURR')[['AMT_ANNUITY']].sum()
mod_prev_app_df['AMT_ANNUITY_prev_count'] = prev_app_df.groupby('SK_ID_CURR')[['AMT_ANNUITY']].count()
mod_prev_app_df['AMT_GOODS_PRICE_prev'] = prev_app_df.groupby('SK_ID_CURR')[['AMT_GOODS_PRICE']].sum()
mod_prev_app_df['AMT_GOODS_PRICE_prev_count'] = prev_app_df.groupby('SK_ID_CURR')[['AMT_GOODS_PRICE']].count()
mod_prev_app_df['AMT_DOWN_PAYMENT_count'] = prev_app_df.groupby('SK_ID_CURR')[['AMT_DOWN_PAYMENT']].count()
mod_prev_app_df['prev_app_count'] = prev_app_df.groupby('SK_ID_CURR')[['SK_ID_PREV']].count()


con_prev_app_df = pd.concat([cat_prev_app_df,mod_prev_app_df], axis = 1)
con_prev_app_df['SK_ID_CURR'] = con_prev_app_df.index
#con_prev_app_df = pd.merge(recent_app_df, cat_prev_app_df, how = 'left', on = 'SK_ID_CURR')
con_prev_app_df = pd.merge(con_prev_app_df, recent_app_df, how = 'left', on = 'SK_ID_CURR')


#mod_prev_app_df.drop(list_col_papp, axis = 1, inplace = True)
#c = a.NAME_CONTRACT_STATUS.get_value(a.DAYS_DECISION[a.DAYS_DECISION == a.DAYS_DECISION.max()].index[0])







# Bureau application 

com_bur_df = pd.merge(bur_df, bur_bal_df, how = 'left', on = 'SK_ID_BUREAU')
act_bur_df = com_bur_df[com_bur_df.CREDIT_ACTIVE == 'Active']
'''
onehot_com_bur_df = pd.get_dummies(data = com_bur_df, columns = ['SK_ID_CURR','CREDIT_TYPE'])
ct_com_bur_df = onehot_com_bur_df.groupby('SK_ID_CURR').sum()
'''
dummy_bur_df = com_bur_df.groupby(['SK_ID_CURR','CREDIT_ACTIVE']).size().reset_index(name = 'count')
dummy_bur_df['CREDIT_Active'] = np.where(dummy_bur_df['CREDIT_ACTIVE'] == 'Active', dummy_bur_df['count'],0)
dummy_bur_df['CREDIT_Closed'] = np.where(dummy_bur_df['CREDIT_ACTIVE'] == 'Closed', dummy_bur_df['count'],0)
dummy_bur_df['CREDIT_Sold'] = np.where(dummy_bur_df['CREDIT_ACTIVE'] == 'Sold', dummy_bur_df['count'],0)
dummy_bur_df['CREDIT_Bad_debt'] = np.where(dummy_bur_df['CREDIT_ACTIVE'] == 'Bad debt', dummy_bur_df['count'],0)
mod_bur_df = dummy_bur_df.groupby('SK_ID_CURR')[['CREDIT_Active','CREDIT_Closed','CREDIT_Sold','CREDIT_Bad_debt']].sum()

# extracting the recent information
idx_recent_bur = com_bur_df.groupby('SK_ID_CURR')['DAYS_CREDIT'].idxmax()
recent_bur_df = pd.DataFrame(com_bur_df.loc[idx_recent_bur].values)
recent_bur_df.columns = list(com_bur_df.columns)
recent_bur_df = recent_bur_df.add_prefix('recent_')
recent_bur_df.rename(columns={'recent_SK_ID_CURR':'SK_ID_CURR'}, inplace=True)
recent_bur_df = recent_bur_df.loc[:, recent_bur_df.columns != 'recent_SK_ID_BUREAU']
recent_bur_df = recent_bur_df.dropna(axis =1)
    


mod_bur_df = pd.concat([mod_bur_df,
                       com_bur_df.groupby('SK_ID_CURR')[['CREDIT_DAY_OVERDUE']].sum(),
                       com_bur_df.groupby('SK_ID_CURR')[['CNT_CREDIT_PROLONG']].sum(),
                       act_bur_df.groupby('SK_ID_CURR')[['AMT_CREDIT_MAX_OVERDUE']].max(),
                       com_bur_df.drop_duplicates(['SK_ID_BUREAU']).groupby('SK_ID_CURR')[['AMT_CREDIT_SUM']].sum(),
                       com_bur_df.drop_duplicates(['SK_ID_BUREAU']).groupby('SK_ID_CURR')[['AMT_CREDIT_SUM_DEBT']].sum(),
                       com_bur_df.drop_duplicates(['SK_ID_BUREAU']).groupby('SK_ID_CURR')[['AMT_CREDIT_SUM_LIMIT']].sum(),
                       com_bur_df.drop_duplicates(['SK_ID_BUREAU']).groupby('SK_ID_CURR')[['AMT_CREDIT_SUM_OVERDUE']].sum()], axis = 1)
                       
mod_bur_df['SK_ID_CURR'] = mod_bur_df.index
mod_bur_df = pd.merge(mod_bur_df, recent_bur_df, how = 'left', on = 'SK_ID_CURR')




mod_bur_df = mod_bur_df.dropna(axis =1)

############################Cleansing is done######################

''' DF to be mergerd
app_df_train
mod_cc_bal_df
pos_cash_bal_df
mod_prev_app_df
inst_df
buereau information



X_df_train = app_df_train.iloc[1:1001,:]
#un_bur_bal_df = bur_bal_df[bur_bal_df.MONTHS_BALANCE == -1]
mer_bal_bur_df = pd.merge(bur_df, bur_bal_df, how = 'left', on = 'SK_ID_BUREAU')
X_df_train = pd.merge(X_df_train, mer_bal_bur_df, how = 'left', on = 'SK_ID_CURR')
#X_df_train.to_csv('Data/X2.csv', sep = ',', encoding='utf-8')

#X_df_train =  pd.read_hdf('store.h5','df')
'''

# train data merge

app_df_train_drop = app_df_train.dropna(thresh = len(app_df_train)*0.8, axis = 1)

#X_df_train = pd.merge(app_df_train, mod_cc_bal_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(app_df_train_drop, pos_cash_bal_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(X_df_train, inst_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(X_df_train, mod_bur_df, how = 'left', on = 'SK_ID_CURR')
X_df_train = pd.merge(X_df_train, con_prev_app_df, how = 'left', on = 'SK_ID_CURR')


#X_df_train.to_csv('Data/X_train.csv')
X_df_train = pd.read_csv('Data/X_train.csv')
X_df_train = X_df_train.loc[:, X_df_train.columns != 'Unnamed: 0']

# data cleansing
'''nans = X_df_train.isnull().sum()
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
'''

# removing the target for Y
#to_remove_col_train = to_remove_col
#to_remove_col_train.append("TARGET")
#thres_col = np.append(thres_col, np.asarray(to_remove_col).reshape(49,1), axis = 1)




X_train = []
X_test = []
y_train = []


# extracting the feature after removing the columns with missing values  more than threshold
#X_df_train = X_df_train.drop(to_remove_col, axis = 1)
X_df_train = X_df_train.dropna(thresh = len(X_df_train)*0.8, axis = 1)
X_df_train = X_df_train.loc[:, X_df_train.columns != 'SK_ID_CURR']


# Auto filling nans with the value forward
cat_col = X_df_train.select_dtypes(include = ['object'])
cat_col1 = X_df_train.select_dtypes(exclude = ['object'])


cat_col = cat_col.fillna(method = 'ffill')
cat_col1 = cat_col1.fillna(cat_col1.mean(), inplace = True)
X_df_train = pd.concat([cat_col,cat_col1], axis = 1)
# label encoding and onehot encoding

X_df_train = pd.get_dummies(X_df_train, drop_first = True)
X_df_train_mod = X_df_train.drop(diff_feat_test, axis = 1)


#X = np.array(X_df_train.iloc[:,2:])
#y = np.array(X_df_train.iloc[:,1])


X = np.array(X_df_train_mod.loc[:])
#X = np.array(X_df_rf.loc[:, X_df_rf.columns != 'TARGET'])
y = X_df_train['TARGET']


# PCA Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_2d = pca.fit_transform(X)

# ploting the 2D
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
plt.scatter(pca_2d[:,0], pca_2d[:,1], c = y)
plt.title('PCA scatter plot')
plt.show()


plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

#choosing number of features required for prediction using PCA
pca.n_features_
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

from sklearn.decomposition import RandomizedPCA
pca1 = RandomizedPCA(40).fit(X)
components = pca1.transform(X)

from sklearn.manifold import TSNE
perplexities = (2, 5, 10, 30, 50, 100)
plt.figure(figsize = (10, 10*len(perplexities)))
for i, perplex in enumerate(perplexities):
    print('perplexity: {}'.format(perplex))
    tsne = TSNE(n_components = 2, perplexity = perplex, n_iter = 1000, verbose = 1)
    tsne_2d = tsne.fit_transform(X)
    
    plt.subplot(int('{}1{}'.format(len(perplexities), i+1)))
    plt.title('t-SNE scatter plot, perplexity = {}'.format(perplex))
    plt.scatter(tsne_2d[:,0], tsne_2d[:,1], c = y)
plt.show()



#a = np.where(pay_df['AMT_INSTALMENT'] == pay_df['AMT_PAYMENT'],1,np.nan)
#normatlization
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




# SMOTE for Neural net
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
sm = SMOTE(ratio = 1.0)
smote_enn = SMOTEENN(smote = sm)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)


# handling unbalanced train
'''
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)'''



from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_train = np.array(sc.fit_transform(X_train))
X_test = sc.transform(X_test)






# Predicting the Test set results



import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# initializing ANN

classifier = Sequential()
# first hidden layer
classifier.add(Dense(units = 11, kernel_initializer = "uniform", activation = "relu", input_dim = X_train_sm.shape[1]))
classifier.add(Dropout(rate = 0.2)) # used not to get over fitting

# second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.2))

# third hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.2))
# output layer

classifier.add(Dense(units = 2, kernel_initializer = "uniform", activation = "sigmoid"))

classifier.summary()
# create ANN
opt = Adam(lr=0.001)
classifier.compile(optimizer = "rmsprop" , loss = "binary_crossentropy", metrics = ["accuracy"])


# fit to training set



history = classifier.fit(x = X_train_sm, y = y_train_sm, batch_size = 64, epochs = 10, validation_split=0.2)

# Improve the classifier/model using grid search

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
#from imblearn.pipeline import make_pipeline
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
def build_classifer(optimizer, k_init, loss):
    np.random.seed(1337)
    classifier = Sequential()
    classifier.add(Dense(units = 25, kernel_initializer = k_init, activation = "relu", input_dim = X_train.shape[1]))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 16, kernel_initializer = k_init, activation = "relu"))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 6, kernel_initializer = k_init, activation = "relu"))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units = 1, kernel_initializer = k_init, activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = loss, metrics = ["accuracy"])
    return classifier
kf = StratifiedKFold(n_splits=10, random_state=111)
classifier = KerasClassifier(build_fn = build_classifer, verbose=0)
#pipeline = make_pipeline(smote_enn, classifier)
pipline = Pipeline([
        ('sampling', SMOTE(random_state=4)),
        ('clf', classifier)
])
parameters = {'clf__batch_size' : [64, 128],
              'clf__epochs' : [20, 40],
              'clf__optimizer' : ['rmsprop', 'adagrad','adam'],
              'clf__k_init': ['glorot_uniform','normal','uniform'],
              'clf__loss' : ['binary_crossentropy' , 'sparse_categorical_crossentropy']}
grid_search = GridSearchCV(pipline,
                           param_grid = parameters,
                           scoring = "recall",
                           cv = kf)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best: %f using %s" % (best_accuracy, best_parameters))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print(datetime.now().time().strftime('%H:%M'))

learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_fraud)

plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()



print(datetime.now().time().strftime('%H:%M'))


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predicting the Test set results
y_pred_fraud = classifier.predict_classes(X_test)


from sklearn.metrics import confusion_matrix
cm_nn = confusion_matrix(y_test, y_pred_fraud)

from sklearn.metrics import roc_auc_score
prob_y_4 = classifier.predict_proba(X_test)
prob_y_4 = [p[1] for p in prob_y_4]
print( roc_auc_score(y_test, prob_y_4) )

#y_pred_fraud = (y_pred_fraud >0.5)
# result = [ round(elem, 0) for elem in y_pred_fraud ]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_fraud)

plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()

# random forest with CV============================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rfmodel = RandomForestClassifier(random_state = 3)
parameters = {'max_depth' : [80 ,100],
              'max_features' : ['auto', 'sqrt'],
              'bootstrap' : [True], 
              'min_samples_leaf': [4, 5],
              'min_samples_split': [10, 12],
              'n_estimators' : [10, 50]}
grid_search = GridSearchCV(estimator = rfmodel,
                           param_grid = parameters,
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


'''sample_leaf = [10,50,100,200,500]
for leaf_size in sample_leaf :
    model = RandomForestClassifier(n_estimators = 200, 
                                  oob_score = True, 
                                  random_state =50, 
                                  max_features = "auto", 
                                  min_samples_leaf = 10)
    model.fit(X_train,y_train)
    print( model.oob_score_, leaf_size )
'''
model = RandomForestClassifier()
model.fit(X_train,y_train)


y_pred_ra = model.predict(X_test)
#y_pred_ra = (y_pred_ra >0.5)
#result = y_pred_ra.astype(int)

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
filename = 'RF_model_FI.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
model = pickle.load(open(filename, 'rb'))
result = model.score(X_test, y_test)
print(result)




# feature importance
import matplotlib.pyplot as plt
plt.plot(0.07,0.07)
features = X_df_train.columns.values[X_df_train.columns != 'TARGET']
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


#plotting the tree from RF

#model.estimators_[0].tree_.feature

from sklearn import tree
from sklearn.tree import export_graphviz
#dotfile = six.StringIO()
i_tree = 0
for tree_in_forest in model.estimators_:
     with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
     i_tree = i_tree + 1



# Gradient boosting machine [classifier]========================================
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.metrics import classification_report

GBM_model = GradientBoostingClassifier(learning_rate=0.1,
                                       n_estimators=100,
                                       max_depth=300,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       subsample=1,
                                       max_features='sqrt',
                                       random_state=10)

GBM_model.fit(X_train,y_train)
predictors=list(X_train)
feat_imp = pd.Series(GBM_model.feature_importances_, features).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')
print('Accuracy of the GBM on test set: {:.3f}'.format(GBM_model.score(X_test, y_test)))
pred=GBM_model.predict(X_test)
print(classification_report(y_test, pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)

# light GBM
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
lgbm_model = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

lgbm_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
               eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

pred=lgbm_model.predict(X_test)
print(classification_report(y_test, pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)

'''


'''
# Kaggle submission set

#Test data merge
app_df_test_drop = app_df_test[list(set(list(app_df_train_drop.columns)) - set(['TARGET']))]
#X_df_test = pd.merge(app_df_test, mod_cc_bal_df, how = 'left', on = 'SK_ID_CURR')
X_df_test = pd.merge(app_df_test_drop, pos_cash_bal_df, how = 'left', on = 'SK_ID_CURR')
X_df_test = pd.merge(X_df_test, inst_df, how = 'left', on = 'SK_ID_CURR')
X_df_test = pd.merge(X_df_test, mod_bur_df, how = 'left', on = 'SK_ID_CURR')
X_df_test = pd.merge(X_df_test, con_prev_app_df, how = 'left', on = 'SK_ID_CURR')


#X_df_test.to_csv('Data/X_test.csv')
X_df_test = pd.read_csv('Data/X_test.csv')
X_df_test = X_df_test.loc[:, X_df_test.columns != 'Unnamed: 0']


X_df_test = X_df_test.loc[:, X_df_test.columns != 'SK_ID_CURR']

#X_df_test = X_df_test.drop(to_remove_col, axis = 1)
# Auto filling nans with the value forward
cat_col_test = X_df_test.select_dtypes(include = ['object'])
cat_col1_test = X_df_test.select_dtypes(exclude = ['object'])


cat_col_test = cat_col_test.fillna(method = 'ffill')
cat_col1_test = cat_col1_test.fillna(cat_col1_test.mean(), inplace = True)
X_df_test = pd.concat([cat_col_test,cat_col1_test], axis = 1)


X_df_test = pd.get_dummies(X_df_test, drop_first = True)

diff_feat_test = list(set(list(X_df_train.columns.values)) - set(list(X_df_test.columns.values)))
diff_feat_train = list(set(list(X_df_test.columns.values)) - set(list(X_df_train.columns.values)))
X_df_test = X_df_test.drop(labels = diff_feat_train, axis =1)
X_df_test1 = X_df_test.reindex(columns=diff_feat_test, fill_value=0)
X_df_test =pd.concat([X_df_test1,X_df_test], axis = 1)
X_df_test_SK = X_df_test[important_feature]
#X_df_test = X_df_test.loc[:, X_df_test.columns != 'NAME_GOODS_CATEGORY_House Construction']

val_set = np.array(X_df_test.loc[:, X_df_test.columns != 'TARGET'])



val_set_scaled = sc.transform(val_set)

X_val = np.array(val_set_scaled)

#y_res = model.predict(X_val)
y_res = classifier.predict_classes(X_val)

'''
y_SK = X_df_test.SK_ID_CURR

kag = np.vstack((list(y_SK),y_res)).T

unique = np.unique(kag[:,0])
kag_sub = []
from collections import Counter

for i in range(len(unique)):
    dum = kag[np.where(kag == unique[i]),:][0]
    count = Counter(dum[:, 1])
    kag_sub.append(count.most_common(1)[0][0])
'''


np.savetxt("Data/pred_nn.csv", y_res, delimiter=",")




#y_pred.tofile('Data/pred.csv', sep = ',')
# dimensionality reduction
'''
cat_col = app_df_train.select_dtypes(include = ['object'])
cat_col = list(app_df_train.select_dtypes(include = ['object']))
cat_col = np.asarray(cat_col)
cat_col = cat_col.reshape((16,1))


# Finding / analysing  Nan

col_nan = data_type.isnull().any()
col_nan = col_nan.reshape((16,1))
col_nan = np.append(col_nan, cat_col, axis = 1)
nan_count = []

for i in range(len(col_nan)):
    nan_count.append(data_type[cat_col[i]].isnull().sum())
  

nan_count = np.asarray(nan_count)
nan_count = nan_count.reshape((16,1))
col_nan = np.append(col_nan, nan_count, axis = 1)
nan_perc = []
for i in range(len(col_nan)):
      nan_perc.append(np.round((nan_count[i].astype(float)/len(data_type)*100), decimals = 2))
nan_perc = np.asarray(nan_perc)
nan_perc = nan_perc.reshape((16,1))
col_nan = np.append(col_nan, nan_perc, axis = 1)



# label encoding and one hot encoding
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = "NaN", strategy = "most_frequent", axis = 1)
data_type['NAME_TYPE_SUITE'] = imp.fit_transform(data_type['NAME_TYPE_SUITE'])

data_type['NAME_TYPE_SUITE'] = data_type.fillna(data_type['NAME_TYPE_SUITE'].value_counts().index[0])


# replace nans with the previous values

data_type1 = data_type.fillna(method = 'ffill')



app_zf_train
pd.get_dummies()'''