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
                "Catboost": CatBoostRegressor(),
                "XGboost": XGBRegressor()            
            }


            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               models=models)
            
            
            ## Find the best model score and model from the dictionary

            best_model_name, best_model_score = max(model_report.items(), key=lambda x:x[1])
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found!")

            save_object(filepath=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                        )
            
            predicted = best_model.predict(X_test)
            r_squared = r2_score(y_test, predicted)

            return r_squared


        except Exception as e:
            CustomException(e,sys)