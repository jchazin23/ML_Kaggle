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

encoded = pd.get_dummies(pd.concat([train_data_x,quiz_data], axis=0))
train_rows = train_data_x.shape[0]

train_encoded_x = encoded.iloc[:train_rows, :]
quiz_encoded = encoded.iloc[train_rows:, :] 



#split data for hold-out

data_train_x = train_encoded_x.loc[0:99999]
data_hold_out_x = train_encoded_x.loc[100000:]
data_train_y = train_data_y.loc[0:99999]
data_hold_out_y = train_data_y.loc[100000:]

#Since we have high-dimensionality, I will start with a Random Forest of Trees
#approach, since its performance is persistant even in high dimensions
#Ensemble --> Random Forests

from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(data_train_x,data_train_y)
#
#hold_out_y_hat = clf.predict(data_hold_out_x)
#score = np.sum(hold_out_y_hat == np.array(data_hold_out_y))/float(len(hold_out_y_hat))
#print score
###0.939002123933
##
##
#quiz_y_hat = pd.DataFrame(clf.predict(quiz_encoded))
#quiz_y_hat.to_csv('quiz_pred_1.csv')

clf2 = RandomForestClassifier(n_estimators = 10)
clf2 = clf2.fit(train_encoded_x,train_data_y)
quiz_y_hat2 = pd.DataFrame(clf2.predict(quiz_encoded))
quiz_y_hat2.to_csv('quiz_pred_2.csv')




    