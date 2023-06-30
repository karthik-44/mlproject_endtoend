import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle

from sklearn.metrics import r2_score, mean_absolute_error as mae, mean_squared_error as mse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(filepath, obj):

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as filehandle:
            pickle.dump(obj, filehandle)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates the models based on the train, test data and provides a metric report for the models
    as a dictionary.
    """
    try:
        report = {}
        best_models = {} #- a dictionary with best performing model from each ml algorithm.
        logging.info("Model Evaluation begun")

        for i in range(len(list(models))):
            logging.info("Model Evaluation for model {}".format(list(models)[i]))

            #get the model and params for the model
            model = list(models.values())[i]
            
            param = params[list(models.keys())[i]]

            # gridsearch
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)          #best hyper-parameter tuned model

            # ## randomsearch
            # rs = RandomizedSearchCV(model, param, cv=3)
            # rs.fit(X_train, y_train)
            # model.set_params(**rs.best_params_)            #best hyper-parameter tuned model


            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

                       
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score

            ## save the best model from each case
            best_models[list(models.keys())[i]] = model

            

        
        logging.info("Model Evaluation completed")

        ## report - contains the best scores obtained from each of the different ml algorithms.
        ## Obtain the best model name, score from report.
        best_model_name, best_model_score = max(report.items(), key=lambda x:x[1]) #the name, score of the model which has the highest score is returned.


        ## best_models - a dictionary with best performing model from each ml algorithm.
        ## extract the best of best performing models and return this along with the report
        the_best_model = best_models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No best model found")
        
        
        logging.info("The best model is {} with a score of {}".format(best_model_name, best_model_score))

        # print('#'*50)
        # print("Report", report, sep='\n')
        # print('#'*50)

        # print('#'*50)
        # print("Best models", best_models, sep='\n')
        # print('#'*50)

        return report, the_best_model

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(filepath):
    try:
        with open(filepath, 'rb') as filehandle:
            return pickle.load(filehandle)
    except Exception as e:
        raise CustomException(e, sys)

