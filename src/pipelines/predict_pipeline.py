import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipelines :
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            print("1. Loading model...")
            model = load_object(file_path=model_path)
            print("2. Model loaded:", type(model))

            print("3. Loading preprocessor...")
            preprocessor = load_object(file_path=preprocessor_path)
            print("4. Preprocessor loaded:", type(preprocessor))

            print("5. Transforming...")
            data_scaled = preprocessor.transform(features)
            print("6. Transform successful")

            print("7. Predicting...")
            preds = model.predict(data_scaled)
            print("8. Prediction successful")

            return preds

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self ,
                gender:str ,
                race_ethnicity : str ,
                parental_level_of_education ,
                lunch : str ,
                test_preparation_course : str,
                reading_score : int ,
                writing_score : int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e , sys)