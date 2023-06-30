import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input array")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors" : KNeighborsRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "XGboost": XGBRegressor()            
            }

            params = {
                    "Random Forest": {
                    #"n_estimators": [10, 50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]                    
                                      },

                    "Decision Tree": {
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                        
                    },

                    "Linear Regression": {
                        
                    },

                    "K-Neighbors": {
                        "n_neighbors": [8,16,32,64,128],
                        "weights": ['uniform', 'distance'],
                        "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
                    },

                    "Gradient Boost": {
                        "learning_rate": [0.01, 0.1, 0.5],
                        #"n_estimators": [8,16,32,64,128],
                        "max_depth": [3, 5, 10],
                        "min_samples_split": [2, 5, 10]
                    },

                    "Adaboost": {
                        "learning_rate": [0.01, 0.1, 0.5],
                        #"n_estimators": [8,16,32,64,128],
                        #"base_estimator": [DecisionTreeRegressor(), RandomForestRegressor()]
                    },

                    "Catboost": {
                        "learning_rate": [0.01, 0.1, 0.5],
                        #"n_estimators": [8,16,32,64,128],
                        "max_depth": [3, 5, 10],
                        "subsample": [0.8, 1.0]
                    },

                    "XGboost": {
                        "learning_rate": [0.01, 0.1, 0.5],
                        #"n_estimators": [8,16,32,64,128],
                        "max_depth": [3, 5, 10],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0]
                    }

                 }

            # models = {
            #     "Random Forest": RandomForestRegressor(),
            #     "Decision Tree": DecisionTreeRegressor(),
            #     "Gradient Boosting": GradientBoostingRegressor(),
            #     #"Linear Regression": LinearRegression(),
            #     "XGBRegressor": XGBRegressor(),
            #     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            #     "AdaBoost Regressor": AdaBoostRegressor(),
            # }

            # params={
            #     "Decision Tree": {
            #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #         # 'splitter':['best','random'],
            #         # 'max_features':['sqrt','log2'],
            #     },
            #     "Random Forest":{
            #         # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
            #         # 'max_features':['sqrt','log2',None],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Gradient Boosting":{
            #         # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            #         'learning_rate':[.1,.01,.05,.001],
            #         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            #         # 'criterion':['squared_error', 'friedman_mse'],
            #         # 'max_features':['auto','sqrt','log2'],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     #"Linear Regression":{},
            #     "XGBRegressor":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "CatBoosting Regressor":{
            #         'depth': [6,8,10],
            #         'learning_rate': [0.01, 0.05, 0.1],
            #         'iterations': [30, 50, 100]
            #     },
            #     "AdaBoost Regressor":{
            #         'learning_rate':[.1,.01,0.5,.001],
            #         # 'loss':['linear','square','exponential'],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
                
            #         "K-Neighbors": {
            #             "n_neighbors": [8,16,32,64,128],
            #             "weights": ['uniform', 'distance'],
            #             "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
            #         }
            # }

            model_report, best_model = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               models=models, params = params)
            
            
            ## Find the best model score and model from the dictionary
            ## If we execute just the below code the best model may be one of the models in models dict, say XGBRegressor() without any
            ## hyper parameter tuning and the model.pkl file will have the model which may not be a better one.
            
            # ##
            # best_model_name, best_model_score = max(model_report.items(), key=lambda x:x[1])
            # best_model = models[best_model_name]
            # ##

            ## So, we have to obtain the best model information after hyper-parameter tuning from the evaluate_models.



            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            
            # logging.info("Best model found!")  ## checks performed in utils.py

            save_object(filepath=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                        )
            
            predicted = best_model.predict(X_test)
            r_squared = r2_score(y_test, predicted)

            logging.info("The best model is {} with a model score of {}. The model is saved to {}".format(
                best_model.__class__.__name__,
                r_squared,
                self.model_trainer_config.trained_model_file_path
            ))

            return r_squared


        except Exception as e:
            CustomException(e,sys)