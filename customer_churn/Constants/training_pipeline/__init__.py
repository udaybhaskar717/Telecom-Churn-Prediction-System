import os

"""
Defining common constant variables for the training pipeline
"""

# Target column for prediction
TARGET_COLUMN = 'Churn'

# Pipeline Configuration
PIPELINE_NAME = 'customer_churn'
ARTIFACT_DIR = 'Artifacts'

# File and Schema Paths
FILE_NAME = 'Telecome_Churn_Data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

# Saved Model Configuration
SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

# Training & Testing File Names (Used in Data Transformation)
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

"""
Data Ingestion Related Constants
"""
DATA_INGESTION_COLLECTION_NAME = "CustomerChurn"
DATA_INGESTION_DATABASE_NAME = "UDAYML"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"

"""
Data Transformation Related Constants
"""
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRAIN_TEST_SPLIT_RATIO = 0.2  # Moved from Data Ingestion
