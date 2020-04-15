# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:50:14 2020
@author: chetanjawlae
"""

import pandas as pd
import numpy as np
df = pd.read_excel(r'\data.xlsx')
#%%
#df.dtypes
# DATA Preprocessing
df.TC_agreed_on.value_counts()

# How much time it takes for a user to move from steps of filling a sign up form !
df['Difference_2to3'] = (pd.to_datetime(df['step3_updated_on'])- pd.to_datetime(df['step2_updated_on'])).apply(lambda x: abs(x / np.timedelta64(1, 's')))

#df.drop(['created_on'],1,inplace=True)

# Company Types - Private Sector // Public Secor
com_typ = df['company_name'].str.findall(r"\bPvt ltd\b|\bpvt limited\b|\bPVT.LTD\b|\bpvt. Ltd\b|\bPrivate limited\b|\bPvt Ltd\b|\bPrivate Limited\b|\bpvt. ltd.\b|\bPVT LTD\b|\bPRIVATE LIMITED\b|\bpvt ltd\b|\bPvt. Ltd\b|\bPvt. Ltd.\b|\bPvt Ltd.\b|\bPvt Limited\b|\bprivate limited\b")
com_typ = pd.DataFrame(com_typ.values)
com_typ[0] = com_typ[0].astype(str)
com_typ[0] = com_typ[0].str.replace(r"[",'')
com_typ[0] = com_typ[0].str.replace(r"]",'')

cm_t = com_typ[0].str.len()
cm_lt = []
for value in cm_t[:]:
    if value > 3:
        cm_lt.append('Pvt Ltd')
    else:
        cm_lt.append(None)

df['Company type'] = cm_lt

df.designation[df.designation == 'DIRECTOR'] = 'Director'
df.designation[df.designation == 'director'] = 'Director'
df.designation[df.designation == 'ceo'] = 'CEO'
df.designation[df.designation == 'Ceo'] = 'CEO'
df.designation[df.designation == 'Sr Manager'] = 'Senior Manager'
df.designation[df.designation == 'MD'] = 'Managing Director'
df.designation[df.designation == 'Proprietor'] = 'Owner'
df.designation[df.designation == 'proprietor'] = 'Owner'
df.designation[df.designation == 'Propriter'] = 'Owner'
df.designation[df.designation == 'system administrator'] = 'System Administrator'
df.designation[df.designation == 'manager'] = 'Manager'
df.designation[df.designation == 'MANAGER'] = 'Manager'
df.designation[df.designation == 'General Manager'] = 'Manager'
df.designation[df.designation == 'Manager IT'] = 'Manager'
df.designation[df.designation == 'Marketing Manager'] = 'Manager'
df.designation[df.designation == 'Operations Manager'] = 'Manager'
df.designation[df.designation == 'Hr Manager'] = 'Manager'
df.designation[df.designation == 'Business Manager'] = 'Manager'
df.designation[df.designation == 'Mangaer'] = 'Manager'
df.designation[df.designation == 'Investment Manager'] = 'Manager'
df.designation[df.designation == 'Associate Manager'] = 'Manager'
df.designation[df.designation == 'Assistant Manager'] = 'Manager'
df.designation[df.designation == 'Zonal Manager - Sales'] = 'Manager'
df.designation[df.designation == 'BDM'] = 'Manager'
df.designation[df.designation == 'GM'] = 'Manager'
df.designation[df.designation == 'DGM'] = 'Manager'
df.designation[df.designation == 'Operation Manager'] = 'Manager'
df.designation[df.designation == 'PM'] = 'Manager'
df.designation[df.designation == 'abm'] = 'Manager'
df.designation[df.designation == 'Senior Manager'] = 'Manager'
df.designation[df.designation == 'Sales Manager'] = 'Manager'
df.designation[df.designation == 'Branch Manager'] = 'Manager'
df.designation[df.designation == 'SENIOR MANAGER'] = 'Manager'
df.designation[df.designation == 'HR/Admin Manager'] = 'HR'
df.designation[df.designation == 'Hr'] = 'HR'
df.designation[df.designation == 'Asst.Manager - Human Resource'] = 'HR'
df.designation[df.designation == 'HR & IT'] = 'HR'
df.designation[df.designation == 'HR & Admin'] = 'HR'
df.designation[df.designation == 'Human Resource Manager'] = 'HR'
df.designation[df.designation == 'Human Resource'] = 'HR'
df.designation[df.designation == 'partner'] = 'Partner'
df.designation[df.designation == 'Co-Founder'] = 'Partner'

# Adress
# address = df['Address'].str.len()

# add_li = []
# for value in address[:]:
#     if value > 15:
#         add_li.append(1)
#     else:
#         add_li.append(0)
# print(add_li)

# df['Is_Address'] = add_li

# Name
name = df['full_name'].str.len()

name_li = []
for value in name[:]:
    if value > 6:
        name_li.append(1)
    else:
        name_li.append(0)
print(name_li)

df['Is_Name'] = name_li

em = df['email_address'].str.findall(r"(?i)\@\w+")
em = pd.DataFrame(em.values.tolist())
em[0] = em[0].astype(str)
em[0] = em[0].str.replace(r"[",'')
em[0] = em[0].str.replace(r"]",'')
em[0]= em[0].str.lower()
em[0].value_counts()

domn = ['@gmail','@yahoo','@hotmail','@outlook','@rediffmail','@live']
    
em[1] = em[0].apply(lambda x : ['@gmail' in x,'@yahoo' in x,
        '@hotmail' in x,'@outlook' in x,'@rediffmail' in x,'@live' in x])

df['E-dom'] = em[1].apply(lambda x : False if True in x else True)    

df['E-dom'] = df['E-dom'].where((pd.notnull(df['E-dom'])), False)

df.drop(['company_name_clean','email_address','mobile_number','full_name','company_name'],1,inplace=True)

df['company_website'] = df['company_website'].apply(lambda x : 1 if pd.notnull(x)==True else 0)


df.drop(['created_on','step2_updated_on','step3_updated_on','TC_agreed_on'],1,inplace=True)


# Convert possible features into categories to simpify analysis
for col in ['country_code', 'designation', 'city', 'country', 'company_website',
       'state', 'industry', 'g_source', 'g_feature', 'Company type','Is_Name', 'E-dom', 'Sold']:
       df[col] = df[col].astype('category')

# Arrange in order
df = df[['country_code', 'designation', 'city', 'country', 'company_website',
       'state', 'industry', 'g_source', 'g_feature', 'Company type',
       'Is_Name', 'E-dom', 'Sold']]

# Label Encoding
colnamele =['country_code', 'designation', 'city', 'country', 'company_website',
       'state', 'industry', 'g_source', 'g_feature', 'Company type',
       'Is_Name', 'E-dom']

# DATA CLEANING FINISHED.
#######################################################################################################################################

# Machine Learning Models : 
#%%
from sklearn import preprocessing
le={}  
#new_df = pd.DataFrame()
for x in colnamele:
    le[x]=preprocessing.LabelEncoder()
for x in colnamele:
    le[x].fit(df[x].astype('str'))
    le_dict_loop = dict(zip(le[x].classes_, le[x].transform(le[x].classes_)))
    df[x] = (df[x].astype('str')).apply(lambda x: le_dict_loop.get(x, -1))
    le[x] = dict(zip(le[x].classes_, le[x].transform(le[x].classes_)))

#%%
X = df.iloc[:,:-1]
y = df.iloc[:,-1] 
#%%
# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 2, shuffle = True, stratify = y)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#knn = KNeighborsClassifier(n_neighbors=17, weights='distance', algorithm='brute', leaf_size=7, p=3)
# =============================================================================
# knn = KNeighborsClassifier(n_neighbors=13, weights='distance', algorithm='brute', leaf_size=9, p=3)
# 
# algorithm='brute', leaf_size=9, metric='minkowski',
#                      metric_params=None, n_jobs=None, n_neighbors=13, p=3,
#                      weights='distance'
# 
# knn.fit(X_train,y_train)
# Y_pred_knn1 = knn.predict(X_test)
# 
# knn_confusion= [confusion_matrix(y_test, Y_pred_knn1)]
# knn_accuracy = [accuracy_score(y_test.tolist(),Y_pred_knn1)]
# print(classification_report(y_test.tolist(),Y_pred_knn1))
# knn_confusion
# knn_accuracy
# =============================================================================

# =============================================================================
#    precision    recall  f1-score   support
# 
#            0       0.98      1.00      0.99      3530
#            1       0.89      0.59      0.71       161
# 
#     accuracy                           0.98      3691
#    macro avg       0.93      0.79      0.85      3691
# weighted avg       0.98      0.98      0.98      3691
# 
# 
# knn_confusion
# Out[17]: 
# [array([[3518,   12],
#         [  66,   95]], dtype=int64)]
# 
# knn_accuracy
# Out[18]: [0.978867515578434]
# 
# =============================================================================

# =============================================================================
# from sklearn.model_selection import GridSearchCV 
# grid_params = { 'n_neighbors': [3,5,11,16,19], 
#                'weights': ['uniform', 'distance'], 
#                'metric':['euclidean', 'manhattan'],
#                'n_jobs':[-1],   
#                'algorithm':['auto','ball_tree','kd_tree','brute'],
#                'leaf_size':[3,6,9,11,30],
#                'p':[3,5,7]}
#                
# gs = GridSearchCV( KNeighborsClassifier(), grid_params, verbose = 1, cv = 3)
# knn = gs.fit(X_train,y_train)
# 
# knn.best_score_
# knn.best_estimator_
# knn.best_params_
# Y_pred_knn1 = knn.best_estimator_.predict(X_test)
# =============================================================================

# =============================================================================

knn = KNeighborsClassifier(algorithm='auto', leaf_size=3, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=16, p=3,
                     weights='distance')

knn = knn.fit(X_train,y_train)
Y_pred_knn1 = knn.predict(X_test)

knn_confusion= [confusion_matrix(y_test, Y_pred_knn1)]
knn_accuracy = [accuracy_score(y_test.tolist(),Y_pred_knn1)]
print(classification_report(y_test.tolist(),Y_pred_knn1))
knn_confusion
knn_accuracy

# =============================================================================
#    precision    recall  f1-score   support
# 
#            0       0.98      1.00      0.99      3530
#            1       0.90      0.60      0.72       161
# 
#     accuracy                           0.98      3691
#    macro avg       0.94      0.80      0.85      3691
# weighted avg       0.98      0.98      0.98      3691
# 
# 
# knn_confusion
# Out[106]: 
# [array([[3519,   11],
#         [  65,   96]], dtype=int64)]
# 
# knn_accuracy
# Out[107]: [0.979409374153346]
# =============================================================================

#%%
#BEST XGB
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bytree=0.8, gamma=0, learning_rate=0.1,
              max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
              n_estimators=200, n_jobs=1, nthread=4,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
              subsample=0.8)

xgb.fit(X_train,y_train)
Y_pred_xgb1 = xgb.predict(X_test)
xgb_confusion= [confusion_matrix(y_test, Y_pred_xgb1)]
xgb_accuracy = [accuracy_score(y_test.tolist(),Y_pred_xgb1)]

print(classification_report(y_test,Y_pred_xgb1))
xgb_confusion
xgb_accuracy

# =============================================================================
#   precision    recall  f1-score   support
# 
#            0       0.97      1.00      0.98      3530
#            1       0.76      0.22      0.34       161
# 
#     accuracy                           0.96      3691
#    macro avg       0.86      0.61      0.66      3691
# weighted avg       0.96      0.96      0.95      3691
# 
# 
# xgb_confusion
# Out[20]: 
# [array([[3519,   11],
#         [ 126,   35]], dtype=int64)]
# 
# xgb_accuracy
# Out[21]: [0.9628826876185316]
# 
# =============================================================================

##############################################################################################################
#######################################################################################################
#######################################################################################################

#%%
#OUTPUT PICKLES

import pickle
pickle.dump(knn, open('diy_knn.pkl', 'wb'))
pickle.dump(xgb, open('diy_xgb.pkl', 'wb'))
np.save('label_encode.npy', le)
pickle.dump(le, open('label_encode', 'wb'))
loaded_model = pickle.load(open('label_encode', 'rb'))
pickle.dump(scaler, open('scaler', 'wb'))
#%%


#%%
#  ENsemble Models OUTPUT
Ensb_data = pd.DataFrame()

Ensb_data['XGB 0'] = 0
Ensb_data['XGB 1'] = 0
Ensb_data['KNN 0'] = 0
Ensb_data['KNN 1'] = 0
Ensb_data['y_test'] = 0

Ensb_data['y_test'] = y_test
Ensb_data[['XGB 0','XGB 1']] = xgb.predict_proba(X_test)
Ensb_data[['KNN 0','KNN 1']] = knn.best_estimator_.predict_proba(X_test)

#Ensb_data.drop(['KNN 0','XGB 0'],1,inplace=True)
#%%
# =============================================================================
# from keras.wrappers.scikit_learn import KerasClassifier
# import keras
# from keras.models import Sequential
# from keras.layers import Dense 
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# 
# 
# def ensb():
#     X_ensb = Ensb_data.iloc[:,:-1]
#     Y_ensb = Ensb_data.iloc[:,-1]
#     x_train_ensb, x_test_ensb, y_train_ensb, y_test_ensb = train_test_split(X_ensb, Y_ensb, test_size=0.30, random_state=5)
#     
#     y_train_ensb = keras.utils.np_utils.to_categorical(y_train_ensb, num_classes=2)
#     y_test_ensb = keras.utils.np_utils.to_categorical(y_test_ensb, num_classes=2)
#     # creating model
#     model = Sequential()
#     model.add(Dense(10, input_dim=4, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(6, activation='relu'))
#     model.add(Dense(6, activation='relu'))
#     model.add(Dense(4, activation='relu'))
#     model.add(Dense(2, activation='relu'))
#     model.add(Dense(2, activation='softmax'))
# 
#     # compile and fit model
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
#     model.fit(x_train_ensb, y_train_ensb, batch_size=15, epochs=25,
#               validation_data=(x_test_ensb, y_test_ensb))
#    dl_out = model.predict(x_test_ensb)
#    
#    d1_out_12 = []
#    
#    for x in dl_out:
#        if x[1] >0.90:
#            d1_out_12.append(1)
#        else:
#            d1_out_12.append(0)
#            
#        
#    dl_confusion= [confusion_matrix(y_test, Y_pred_xgb1)]
#        
# =============================================================================


#%%
# Balancing Probabilities based on their weights
# Rearranging Opportunities in descending order. Best Possibility to least possibility cases.

Prediction = Ensb_data.copy()
import sklearn
for i in Prediction[['KNN 1','XGB 1']].columns:
    Prediction[i]=sklearn.preprocessing.minmax_scale(Prediction[i], feature_range=(0,1), axis=0, copy=True)

Prediction.sort_values(['KNN 1'],ascending=False,inplace=True)
Prediction1 = Prediction[Prediction['KNN 1'] !=0]
Prediction1['Probability'] = Prediction1['KNN 1']
Prediction2 = Prediction[Prediction['KNN 1'] ==0].sort_values(['XGB 1'],ascending=False)
Prediction2['Probability'] = Prediction2['XGB 1']
Prediction_final = pd.concat([Prediction1,Prediction2]).reset_index(drop=True)
Prediction_final['Date'] = strtdate

#print(Prediction_final)

# Updating data to server
engine = create_engine("")
conn = engine.connect()
Prediction_final.to_sql('Sales_Probability',con=conn,if_exists='append',index = False)
conn.close()