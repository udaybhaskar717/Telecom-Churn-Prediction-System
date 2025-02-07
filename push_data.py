import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging

class CustomerChurnDataExtract:
    def __init__(self, FILE_PATH, DATABASE, Collection, MONGO_DB_URL=MONGO_DB_URL):
        try:
            logging.info("Initializing CustomerChurnDataExtract class.")
            self.FILE_PATH = FILE_PATH
            self.DATABASE = DATABASE
            self.Collection = Collection
            self.MONGO_DB_URL = MONGO_DB_URL
            logging.info(f"Initialized with FILE_PATH: {FILE_PATH}, DATABASE: {DATABASE}, Collection: {Collection}")
        except Exception as e:
            logging.error(f"Error occurred during initialization: {e}")
            raise CustomerChurnException(e, sys)

    def csv_to_json_convert(self):
        try:
            logging.info(f"Reading CSV file from path: {self.FILE_PATH}")
            data = pd.read_csv(self.FILE_PATH)
            data.reset_index(drop=True, inplace=True)
            logging.info("CSV file read successfully. Converting to JSON format.")
            records = list(json.loads(data.T.to_json()).values())
            logging.info(f"Successfully converted {len(records)} records to JSON.")
        except Exception as e:
            logging.error(f"Error occurred during CSV to JSON conversion: {e}")
            raise CustomerChurnException(e, sys)
        return records

    def insert_data_mongodb(self, records):
        try:
            logging.info("Connecting to MongoDB.")
            self.mongo_client = pymongo.MongoClient(self.MONGO_DB_URL, tlsCAFile=ca)
            self.db = self.mongo_client[self.DATABASE]
            self.collection = self.db[self.Collection]

            # Clear the existing data in the collection before inserting new records
            logging.info(f"Clearing existing data in collection: {self.Collection}")
            self.collection.delete_many({})  # This clears all data in the collection
            logging.info("Existing data cleared successfully.")

            logging.info(f"Inserting {len(records)} records into collection: {self.Collection}.")
            self.collection.insert_many(records)
            logging.info(f"Successfully inserted {len(records)} records into MongoDB.")
        except Exception as e:
            logging.error(f"Error occurred while inserting data into MongoDB: {e}")
            raise CustomerChurnException(e, sys)
        return len(records)

if __name__ == "__main__":
    try:
        logging.info("Starting the Customer Churn Data Extraction process.")
        FILE_PATH = 'Telecome_Churn_Data\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
        DATABASE = "UDAYML"
        Collection = "CustomerChurn"
        logging.info(f"FILE_PATH: {FILE_PATH}, DATABASE: {DATABASE}, Collection: {Collection}")

        customerChurnObj = CustomerChurnDataExtract(FILE_PATH, DATABASE, Collection)
        records = customerChurnObj.csv_to_json_convert()
        no_of_records = customerChurnObj.insert_data_mongodb(records=records)
        logging.info(f"Process completed successfully. Total records inserted: {no_of_records}")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
