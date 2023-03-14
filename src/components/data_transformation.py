import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            numeric_columns = ["writing score","reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                    
                ]
            )
            logging.info("Numerical columns scaling completed")
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numeric_columns),
                    ("cat_pipeline",categorical_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data has completed")
            logging.info("obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_obj()
            target_column_name = "math score"
            numeric_columns = ["writing_score","reading_score"]
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)