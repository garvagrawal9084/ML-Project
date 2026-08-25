import os
import sys

import numpy as np
import pandas as pd
import dill
from matplotlib.dates import drange
from scipy.special.cython_special import cbrt
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV


def save_object(file_path , obj) :
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , "wb") as file_obj :
            dill.dump(obj , file_obj)

    except Exception as e :
        raise CustomException(e , sys)


def evaluate_models(x_train, y_train, x_test, y_test, models: dict, params: dict):
    try:
        report = {}
        trained_models = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            rs = RandomizedSearchCV(
                model,
                para,
                cv=5,
                n_jobs=-1,
                verbose=3,
                refit=True
            )

            rs.fit(x_train, y_train)

            best_model = rs.best_estimator_

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            # Store the fitted model
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e , sys)
