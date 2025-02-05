import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from dotenv import load_dotenv
load_dotenv()

from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging
from customer_churn.entity.config_entity import DataIngestionConfig

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
            df = pd.DataFrame(list(Collection.find()))  # ✅ Fixed this issue

            if df.empty:
                logging.warning("No data found in MongoDB collection.")

            if "_id" in df.columns.to_list():
                logging.info("Dropping '_id' column from DataFrame")
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"Exported {df.shape[0]} rows from MongoDB successfully.")
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

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process: Fetch from MongoDB → Save as CSV
        """
        try:
            logging.info("Starting data ingestion process...")

            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)

            logging.info("Data ingestion completed successfully.")
            return dataframe  # ✅ Return the DataFrame for further processing
        except Exception as e:
            raise CustomerChurnException(e, sys)
