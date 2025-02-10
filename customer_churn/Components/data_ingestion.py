import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from dotenv import load_dotenv
load_dotenv()

from customer_churn.Logging.logger import logging
from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.entity.config_entity import DataIngestionConfig
from customer_churn.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info(f"Initialized Data Ingestion with config: {data_ingestion_config}")
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Fetches raw data from MongoDB and converts it into a Pandas DataFrame.
        """
        try:
            logging.info("Connecting to MongoDB...")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            Collection = mongo_client[database_name][collection_name]

            logging.info(f"Fetching data from MongoDB: Database = {database_name}, Collection = {collection_name}")
            df = pd.DataFrame(list(Collection.find())) 

            if df.empty:
                logging.warning("No data found in MongoDB collection.")

            if "_id" in df.columns.to_list():
                logging.info("Dropping '_id' column from DataFrame")
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Saves the raw data as a CSV file in the Feature Store.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            logging.info(f"Creating feature store directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving raw data to feature store at: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info("Data successfully saved in feature store.")
            return dataframe
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the dataset into training and testing sets.
        """
        try:
            logging.info("Performing train-test split on the dataframe...")
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info(f"Train-test split completed. Train size: {train_set.shape[0]}, Test size: {test_set.shape[0]}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            logging.info(f"Creating directory for train/test files: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train file to: {self.data_ingestion_config.training_file_path}")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            logging.info(f"Exporting test file to: {self.data_ingestion_config.testing_file_path}")
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info("Train and test files exported successfully.")
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process: Fetch from MongoDB → Save as CSV → Split into Train/Test.
        """
        try:
            logging.info("Starting data ingestion process...")

            logging.info("Fetching data from MongoDB...")
            dataframe = self.export_collection_as_dataframe()

            logging.info("Saving raw data to feature store...")
            dataframe = self.export_data_into_feature_store(dataframe)

            logging.info("Splitting data into train and test sets...")
            self.split_data_as_train_test(dataframe)

            logging.info("Creating DataIngestionArtifact...")
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion process completed successfully.")
            return dataingestionartifact

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomerChurnException(e, sys)