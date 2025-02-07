from customer_churn.Constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os
import sys

from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging

class ChurnModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CustomerChurnException(e,sys)
