from customer_churn.Components.data_ingestion import DataIngestion
from customer_churn.Components.data_validation import DataValidation
from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging
from customer_churn.entity.config_entity import DataIngestionConfig, DataValidationConfig,DataTransformationConfig
from customer_churn.entity.config_entity import TrainingPipelineConfig
from customer_churn.Components.data_transformation import DataTransformation
from customer_churn.entity.config_entity import ModelTrainerConfig
from customer_churn.Components.model_trainer import ModelTrainer

import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()  # Corrected variable name
        logging.info("Data Ingestion Completed")
        print(dataingestionartifact)

        # Data Validation
        datavalidationconfig = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, datavalidationconfig)  # Corrected variable name
        logging.info("Initiate data validation")
        datavalidationartifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(datavalidationartifact)
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(datavalidationartifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")

        # logging.info("Model Training sstared")
        # model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        # model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        # model_trainer_artifact=model_trainer.initiate_model_trainer()
        # logging.info("Model Training artifact created")
        
    except Exception as e:
        raise CustomerChurnException(e, sys)
