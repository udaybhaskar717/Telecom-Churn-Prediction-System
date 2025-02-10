import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from customer_churn.Constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from customer_churn.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from customer_churn.entity.config_entity import DataTransformationConfig
from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging
from customer_churn.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            logging.info("Initializing DataTransformation class")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logging.info("DataTransformation class initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing DataTransformation class: {e}")
            raise CustomerChurnException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"Data read successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise CustomerChurnException(e, sys)
    
    @staticmethod
    def replace_empty_with_nan(X):
        """Replace empty strings in 'TotalCharges' column with NaN and convert to float."""
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            if 'TotalCharges' in X.columns:
                X['TotalCharges'] = X['TotalCharges'].replace(" ", np.nan).astype(float)
        return X

    def get_data_transformer_object(self) -> ColumnTransformer:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_features = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
            
            logging.info(f"Identified numerical features: {numerical_features}")
            logging.info(f"Identified categorical features: {categorical_features}")

            # Custom preprocessing step
            custom_preprocessor = FunctionTransformer(self.replace_empty_with_nan)

            # Pipeline for numerical features
            num_pipeline = Pipeline([
                ("custom_preprocessor", custom_preprocessor),  # Step to replace " " with NaN
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])
            logging.info("Numerical pipeline created with custom preprocessing, KNNImputer, and StandardScaler")

            # Pipeline for categorical features
            cat_pipeline = Pipeline([
                ("encoder", OneHotEncoder(handle_unknown='ignore'))
            ])
            logging.info("Categorical pipeline created with OneHotEncoder")

            # Column transformer to apply transformations
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])
            logging.info("ColumnTransformer created with custom preprocessor, numerical, and categorical pipelines")

            logging.info("Exiting get_data_transformer_object method of DataTransformation class")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object method: {e}")
            raise CustomerChurnException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation process")

            # Read training and testing data
            logging.info("Reading training and testing data")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            logging.info(f"Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}")

            # Splitting input and target features
            logging.info("Splitting input and target features")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace({'No': 0, 'Yes': 1})

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace({'No': 0, 'Yes': 1})
            logging.info("Input and target features split successfully")

            # Get the preprocessor object
            logging.info("Getting the preprocessor object")
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            logging.info("Preprocessor object fitted successfully")

            # Transform the input features
            logging.info("Transforming input features")
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            logging.info("Input features transformed successfully")

            logging.info("Applying SMOTE Over sampling...")
            smote = SMOTE()
            transformed_input_train_feature, target_feature_train_df = smote.fit_resample(
                transformed_input_train_feature, target_feature_train_df)
            logging.info("SMOTE Over sampling is applied")

            # Combine transformed features with target features
            logging.info("Combining transformed features with target features")
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            logging.info("Transformed data combined with target features")

            # Save transformed data
            logging.info("Saving transformed data")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Transformed data saved successfully")

            # Save preprocessor object
            logging.info("Saving preprocessor object")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            logging.info("Preprocessor object saved successfully")

            # Prepare data transformation artifact
            logging.info("Preparing DataTransformationArtifact")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("DataTransformationArtifact prepared successfully")

            logging.info("Exiting initiate_data_transformation method of DataTransformation class")
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error in initiate_data_transformation method: {e}")
            raise CustomerChurnException(e, sys)
