#Importing necessary libraries
import pandas as pd
import numpy as np
import string
import csv
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from fuzzywuzzy import fuzz
import pandas_ml as pdml
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
import pickle


#Cleaning null values of TF-IDF and lemmatizated features
class DataCleaning:
    
    #Cleaning the first half of training dataset
    def clean_data(self, X):
        X['tfidf_word_match'] = X['tfidf_word_match'].fillna(0)
        X['lem_tfidf_word_match'] = X['lem_tfidf_word_match'].fillna(0)
        X['log_lem_tfidf'] = X['log_lem_tfidf'].fillna(0)
        X['lem_tfidf_squared'] = X['lem_tfidf_squared'].fillna(0)
        X['lem_tfidf_sqrt'] = X['lem_tfidf_sqrt'].fillna(0)


#Parameter Tuning for XGBoost based on the dataset
class ParameterTuning:
    
    def __init__(self, X, Y):
        self.get_best_learning_rate(X, Y)
        self.get_best_depth_weight(X, Y)
        
        
    #Best learning rate 0.3 for the first XGBoost    
    def get_best_learning_rate(self, X, Y):
    
        n = Y.shape
        Y=np.array(Y).reshape((n[0],))

        model = XGBClassifier()
        learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        param_grid = dict(learning_rate=learning_rate)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(X, Y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
        #Best max depth 5 and min_child_weight 3 for the first XGBoost
        def get_best_depth_weight(self, X, Y):

            n = Y.shape
            Y=np.array(Y).reshape((n[0],))

            param_test1 = {
             'max_depth':range(3,10,1),
             'min_child_weight':range(1,6,1)
            }

            gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.3, n_estimators=140, max_depth=5,
             min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
             objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
             param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

            gsearch1.fit(X,Y)
            gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


#Multi Layer Model Training using Stacking
#Layer 1 - Random Forest + XGBoost
#Layer2 - XGBoost
class ModelTraining:
    
    xgbModel = XGBClassifier()
    randomModel = RandomForestClassifier()
    stackedModel = XGBClassifier()
        
    def layer_one_model_training(self, X, Y, val, test_X):
        
        #Training a Random Forest Model based on training data set
        model = RandomForestClassifier(
            n_estimators=50,
            n_jobs=8)
        self.randomModel = model.fit(X, Y)
        
        #Training a XGB Model based on training data set
        #Parameters used are the ones we got from parameter tuning
        #obj = ParameterTuning(X, Y)
        model1 = XGBClassifier(
         learning_rate=0.3,
         n_estimators=140,
         max_depth=5,
         min_child_weight=3,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)
        self.xgbModel = model1.fit(X, Y, verbose=True)


        #Forest Values for validation and test data set
        X2 = val.iloc[0:,0:43]
        Y2 = val.iloc[0:,43:44]
        
        forest_val_value = model.predict_proba(X2)[:, 1]
        
        #XGB Values for validation and test data set
        xgb_val_value = model1.predict_proba(X2)[:, 1]
        self.layer_two_model_training(forest_val_value, xgb_val_value, forest_test_value, xgb_test_value, Y2, test_X)
        
        
    def layer_two_model_training(self, forest_val_value, xgb_val_value, forest_test_value, xgb_test_value, Y2, test_X):
                
        #Creating a new data frame for Validation Dataset with two features
        #It's predicted Random Forest and XGB values
        X1 = pd.DataFrame({'forest_value': forest_val_value, 'xgb_value': xgb_val_value})
        
        #Training a XGB Model based on this new validation data frame
        #Parameter values used are after tuning them accordingly
        #obj = ParameterTuning(X1, Y1)
        
        model2 = XGBClassifier(
         learning_rate =0.1,
         n_estimators=1000,
         max_depth=2,
         min_child_weight=6,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)

        self.stackedModel = model2.fit(X1, Y2, verbose=True)
        


class StartingClass:
    
    def __init__(self):
        
        #Read the Featured Engineering Files
        data = pd.read_csv('../FeatureEngineeringFiles/featured_train3.csv', engine='python')
        test_X = pd.read_csv('../FeatureEngineeringFiles/featured_test3.csv', engine='python')
        
        #Cleaning the dataset of null values
        cleaning = DataCleaning()
        cleaning.clean_data(data)
        cleaning.clean_data(test_X)
        
        #Dividing data set in training and validation
        #Tried combinations of splits, 0.4 worked the best
        train, val = train_test_split(data, train_size=0.4)
        train = pd.DataFrame(train)
        val = pd.DataFrame(val)
        
        #X contains training without is_duplicate
        #Y contains the target column is_duplicate
        X = train.iloc[0:,0:43]
        Y = train.iloc[0:,43:44]
        
        #Model Training
        model = ModelTraining()
        model.layer_one_model_training(X, Y, val, test_X)
        
        
        
        pickle.dump(model.xgbModel, open("../PickleFiles/xgbModel.pkl", 'wb'))
        pickle.dump(model.randomModel, open("../PickleFiles/randomModel.pkl", 'wb'))
        pickle.dump(model.stackedModel, open("../PickleFiles/stackModel.pkl", 'wb'))
        
 
#Starting Point
if __name__ == '__main__':
    #Read csv files
    obj = StartingClass()
