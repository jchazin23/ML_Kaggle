# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:45:32 2016

@author: jchazin
"""
import numpy as np
import pandas as pd


train_data = pd.read_csv('data.csv',sep=',')
quiz_data = pd.read_csv('quiz.csv',sep=',')

train_data_x = train_data.iloc[:,:-1]
train_data_y = train_data.iloc[:,-1]

## Before running the prediction, we need to homogenize the columns 
## of quiz, since the OneHot procedure only produces columns for unique
## categorical values in the original dataset. For instance, for column '5',
## the training data has an instance of u'5_acomp', but the quiz data does not.
## To handle this, I will loop through and insert columns of zeroes. Since 
## these features are generally so sparse, I don't anticipate this causing 
## much trouble.

## After researching, it seems reasonable instead to combine the 
## test and training data, then run get_dummies to do one-hot encoding
## so as to avoid having column mis-matches. This should only add 
## 338 extra columns to the model, as 
## len(set(quiz.columns)-set(data.columns)) == 338


## how about some feature expansion on the non-categorical data?
## Going to re-order dataframe, split out non-categorical features, 
## then do degree = 2 feature expansion, then re-join, then do 
## one-hot on categorical features. 

#cols=train_data_x.columns.tolist()
#colTypes = train_data_x.dtypes
#colFloats = np.where(colTypes != 'object')
#colObjects = np.where(colTypes == 'object')
#colFloats = colFloats[0].tolist()
#colObjects = colObjects[0].tolist()
##This builds a new list of columns ordered (Floats...Objects)
#colFloats.extend(colObjects)
#52 Feature columns
#cols[0:35 == Float; 36:51 == Object]
#train_data_x = train_data_x[colFloats]
#quiz_data = quiz_data[colFloats]
##now data is re-ordered, time to split out Floats and do feature expansion
#train_data_x_floats = train_data_x.iloc[:,0:36]
#train_data_x_objects = train_data_x.iloc[:,36:]
#quiz_data_floats = quiz_data.iloc[:,0:36]
#quiz_data_objects = quiz_data.iloc[:,36:]

#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(2)
#train_data_x_floats_expanded = pd.DataFrame(poly.fit_transform(train_data_x_floats))
#quiz_data_floats_expanded = pd.DataFrame(poly.fit_transform(quiz_data_floats))
#
#train_data_x = pd.concat([train_data_x_floats_expanded,train_data_x_objects],axis=1)
#quiz_data = pd.concat([quiz_data_floats_expanded,quiz_data_objects],axis=1)


#Here's where we do one-hot encoding of categorical variables
encoded = pd.get_dummies(pd.concat([train_data_x,quiz_data], axis=0))
train_rows = train_data_x.shape[0]

train_encoded_x = encoded.iloc[:train_rows, :]
quiz_encoded = encoded.iloc[train_rows:, :] 



#split data for hold-out
#
#data_train_x = train_encoded_x.loc[0:99999]
#data_hold_out_x = train_encoded_x.loc[100000:]
#data_train_y = train_data_y.loc[0:99999]
#data_hold_out_y = train_data_y.loc[100000:]

#Since we have high-dimensionality, I will start with a Random Forest of Trees
#approach, since its performance is persistant even in high dimensions
#Ensemble --> Random Forests

from sklearn.ensemble import RandomForestClassifier
##Best score so far seems to be at n = 52 for number of estimators
##Extremely diminishing returns thereafter
clf2 = RandomForestClassifier(n_estimators = 64)
clf2 = clf2.fit(train_encoded_x,train_data_y)
quiz_y_hat2 = pd.DataFrame(clf2.predict(quiz_encoded))
quiz_y_hat2.to_csv('quiz_pred_64_final.csv')


#n_estimators_scores = {}
#
#k_estimators = [5,10,15,20,25,30,35,40]
#
#for i in range(len(k_estimators)):
#    clf = RandomForestClassifier(n_estimators=k_estimators[i])
#    clf = clf.fit(data_train_x,data_train_y)
#    hold_out_y_hat = clf.predict(data_hold_out_x)
#    score = np.sum(hold_out_y_hat == np.array(data_hold_out_y))/float(len(hold_out_y_hat))
#    n_estimators_scores[k_estimators[i]] = score
#
#print n_estimators_scores
###40 estimators = 0.94410701643253714
##
##
#quiz_y_hat = pd.DataFrame(clf.predict(quiz_encoded))
#quiz_y_hat.to_csv('quiz_pred_1.csv')








    