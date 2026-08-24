import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models

from src.utils import save_object

@dataclass()
class ModelTrainerConfig :
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_arr , test_arr) :
        try:
            logging.info("Split training and test input data")
            X_train , y_train , X_test , y_test = (
                train_arr[: , :-1] ,
                train_arr[: , -1] ,
                test_arr[: , :-1] ,
                test_arr[: , -1]
            )

            models = {
                "Linear Regression" : LinearRegression() ,
                "Ridge" : Ridge() ,
                "Lasso" : Lasso() ,
                "K-Neighbour Regression" : KNeighborsRegressor() ,
                "Random Forest" : RandomForestRegressor() ,
                "Decision Tree" : DecisionTreeRegressor() ,
                "Gradient Boosting" : GradientBoostingRegressor() ,
                "CatBoosting Regression" : CatBoostRegressor(verbose=False) ,
                "AdaBoost Regression" : AdaBoostRegressor() ,
                "XGBRegression" : XGBRegressor()
            }

            params = {

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                    "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
                },

                "Lasso": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                    "fit_intercept": [True, False],
                    "selection": ["cyclic", "random"]
                },

                "K-Neighbour Regression": {
                    "n_neighbors": [3, 5, 7, 10, 15, 20],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]
                },

                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.8, 0.9, 1.0]
                },

                "CatBoosting Regression": {
                    "iterations": [100, 200, 500],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 10]
                },

                "AdaBoost Regression": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"]
                },

                "XGBRegression": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.2]
                }
            }

            model_report : dict = evaluate_models(x_train = X_train , y_train = y_train
                                            , x_test= X_test , y_test = y_test , models = models , params = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = max(model_report , key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset {best_model_name} {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path ,
                obj=best_model
            )

            return best_model_score


        except Exception as e:
            raise CustomException(e , sys)