import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle

from sklearn.metrics import r2_score, mean_absolute_error as mae, mean_squared_error as mse

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

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates the models based on the train, test data and provides a metric report for the models
    as a dictionary.
    """
    try:
        report = {}
        logging.info("Model Evaluation begun")

        for i in range(len(list(models))):
            logging.info("Model Evaluation for model {}".format(list(models)[i]))

            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        logging.info("Model Evaluation completed")

        return report

    except Exception as e:
        raise CustomException(e,sys)