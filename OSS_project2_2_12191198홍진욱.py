#!/usr/bin/env python
# coding: utf-8

# In[140]:


import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

def sort_dataset(dataset_df):   
    
    return dataset_df.sort_values('year',ascending=True)

def split_dataset(dataset_df):	   
    X_train = dataset_df.drop(columns='salary', axis=1).iloc[:1718]
    X_test = dataset_df.drop(columns='salary', axis=1).iloc[1718:]

    Y_train = dataset_df['salary'].iloc[:1718]*0.001
    Y_test = dataset_df['salary'].iloc[1718:]*0.001 
    
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    
    return dataset_df.loc[:,['age','G','PA','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','HBP','SO','GDP','fly','war']]

def train_predict_decision_tree(X_train, Y_train, X_test):   
    dtree_cls=DecisionTreeRegressor()
    dtree_cls.fit(X_train,Y_train)
   
    return dtree_cls.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):   
    rforest_cls=RandomForestRegressor()
    rforest_cls.fit(X_train,Y_train)  
    
    return rforest_cls.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):    
    svm_pipe=make_pipeline(StandardScaler(),SVR())  
    svm_pipe.fit(X_train, Y_train)
    
    return svm_pipe.predict(X_test)

def calculate_RMSE(labels, predictions):
    
    return np.sqrt(np.mean((predictions-labels)**2))

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))


# In[ ]:





# In[ ]:




