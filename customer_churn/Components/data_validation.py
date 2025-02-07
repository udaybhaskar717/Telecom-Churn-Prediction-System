from customer_churn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from customer_churn.entity.config_entity import DataValidationConfig
from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging
from customer_churn.Constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from customer_churn.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            logging.info("Initializing DataValidation component.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            logging.info(f"Reading schema configuration from: {SCHEMA_FILE_PATH}")
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("Schema configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error during initialization of DataValidation: {e}")
            raise CustomerChurnException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise CustomerChurnException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating number of columns in the dataset.")
            expected_columns = self._schema_config['columns'].keys()
            if set(dataframe.columns) == set(expected_columns):
                logging.info("Column validation successful.")
                return True
            logging.error(f"Column mismatch. Expected: {expected_columns}, Found: {dataframe.columns}")
            return False
        except Exception as e:
            logging.error(f"Error during column validation: {e}")
            raise CustomerChurnException(e, sys)

    def validate_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating missing values in the dataset.")
            if dataframe.isnull().sum().sum() > 0:
                logging.error("Dataset contains missing values.")
                return False
            logging.info("No missing values found.")
            return True
        except Exception as e:
            logging.error(f"Error during missing values validation: {e}")
            raise CustomerChurnException(e, sys)

    def validate_unique_customer_id(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating uniqueness of customerID in the dataset.")
            if dataframe['customerID'].duplicated().sum() > 0:
                logging.error("Duplicate customerID values found.")
                return False
            logging.info("All customerID values are unique.")
            return True
        except Exception as e:
            logging.error(f"Error during customerID uniqueness validation: {e}")
            raise CustomerChurnException(e, sys)

    def validate_categorical_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating categorical values in the dataset.")
            for col, allowed_values in self._schema_config['categorical_columns'].items():
                if not dataframe[col].isin(allowed_values).all():
                    logging.error(f"Unexpected values found in {col}.")
                    return False
            logging.info("All categorical values are valid.")
            return True
        except Exception as e:
            logging.error(f"Error during categorical values validation: {e}")
            raise CustomerChurnException(e, sys)

    def detect_outliers(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Detecting outliers in numerical columns.")
            for col in self._schema_config['numerical_columns']:
                Q1 = dataframe[col].quantile(0.25)
                Q3 = dataframe[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = dataframe[(dataframe[col] < (Q1 - 1.5 * IQR)) | (dataframe[col] > (Q3 + 1.5 * IQR))]
                if not outliers.empty:
                    logging.warning(f"Outliers detected in {col}.")
            logging.info("Outlier detection completed.")
            return True
        except Exception as e:
            logging.error(f"Error during outlier detection: {e}")
            raise CustomerChurnException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            logging.info("Detecting dataset drift between base and current datasets.")
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving drift report to: {drift_report_file_path}")
            write_yaml_file(filepath=drift_report_file_path, content=report)
            logging.info(f"Dataset drift detection completed. Drift status: {status}")
            return status
        except Exception as e:
            logging.error(f"Error during dataset drift detection: {e}")
            raise CustomerChurnException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process.")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Training and testing datasets loaded successfully.")

            if not self.validate_number_of_columns(train_df) or not self.validate_number_of_columns(test_df):
                logging.error("Column validation failed.")
                raise CustomerChurnException("Column validation failed.", sys)
            if not self.validate_missing_values(train_df) or not self.validate_missing_values(test_df):
                logging.error("Missing values found in the dataset.")
                raise CustomerChurnException("Missing values found.", sys)
            if not self.validate_unique_customer_id(train_df):
                logging.error("Duplicate customerID found in training data.")
                raise CustomerChurnException("Duplicate customerID found in training data.", sys)
            if not self.validate_categorical_values(train_df):
                logging.error("Categorical values mismatch.")
                raise CustomerChurnException("Categorical values mismatch.", sys)

            logging.info("Checking for dataset drift between training and testing datasets.")
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Directory created for validated data: {dir_path}")

            logging.info("Saving validated training and testing datasets.")
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
            logging.info("Validated datasets saved successfully.")

            return DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            logging.error(f"Error during data validation process: {e}")
            raise CustomerChurnException(e, sys)