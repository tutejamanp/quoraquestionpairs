import pandas as pd
import numpy as np
import string
import csv   
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
import lightgbm as lgbm
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
        

class TestingPickle:
    
    #Testing the pickle file using it's constructor
    def __init__(self):
            
       
        test_featured = pd.read_csv('../FeatureEngineeringFiles/featured_test.csv', engine='python')
        
        xgbModel = pickle.load(open("../PickleFiles/xgbModel.pkl",'rb'))
        randomModel = pickle.load(open("../PickleFiles/randomModel.pkl",'rb'))
        stackedModel = pickle.load(open("../PickleFiles/stackModel.pkl",'rb'))

        
        test_X = pd.DataFrame()
        test_X['forest_value'] = randomModel.predict_proba(test_featured)[:, 1]
        test_X['xgb_value'] = xgbModel.predict_proba(test_featured)[:, 1]
        p = stackedModel.predict_proba(test_X)[:, 1]
        
                
        #Creating a data set for test data set
        #Predicting values for that
        test_X = pd.read_csv('../InputFiles/test.csv', engine='python')
        
        sub = pd.DataFrame({'id': test_X['test_id'], 'is_duplicate': p})
        sub.to_csv('../OutputFiles/pickled_output.csv', index=False)


class StartClass:
    
    if __name__ == '__main__':
        #Testing the various pickle models and stacking them together to get the final answer
        #Just need to import all the necessary libraries
        #Load the Pickler Class
        #Next load the TestingPickle Class
        testing_pickle = TestingPickle()
