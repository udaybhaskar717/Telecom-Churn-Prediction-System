import os
import sys

from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging

from customer_churn.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from customer_churn.entity.config_entity import ModelTrainerConfig

from customer_churn.utils.ml_utils.model.estimator import ChurnModel
from customer_churn.utils.main_utils.utils import save_object,load_object
from customer_churn.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from customer_churn.utils.ml_utils.metric.classification_report import get_classification_score

from  sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
from urllib.parse import urlparse

import dagshub


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/udaybhaskar717/Telecom-Churn-Prediction-System.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "udaybhaskar717"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "1bc5be36a28d9d43130088a69b9ada150bb4d013"

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def track_mlflow(self,best_model,classficationmetric):
        mlflow.set_registry_uri("https://dagshub.com/udaybhaskar717/Telecom-Churn-Prediction-System.mlflow")
        tracking_url_type_score= urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score = classficationmetric.f1_score
            precision_score = classficationmetric.precision_score
            recall_score = classficationmetric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precesion",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")

            if tracking_url_type_score != "file":
                # mlflow.sklearn.log_model(best_model,"model",registered_model_name=best_model)
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=type(best_model).__name__)

            else:
                mlflow.sklearn.log_model(best_model,"model")

    def train_model(self,X_train,y_train,x_test,y_test):
            
            
            
            models = {
                "RandomForest": RandomForestClassifier(verbose=1),
                "GradientBoosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "LogisticRegression": LogisticRegression()
            }
            param_grids = {
                "RandomForest": {'n_estimators': [100, 200], 'max_depth': [10, 20]},
                "GradientBoosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
                "AdaBoost": {'n_estimators': [50, 100]},
                "DecisionTree": {'max_depth': [5, 10, 20]},
                "KNN": {'n_neighbors': [3, 5, 7]},
                "LogisticRegression": {'C': [0.1, 1, 10]}
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=param_grids)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            y_train_pred=best_model.predict(X_train)

            classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
            
            ## Track the experiements with mlflow
            self.track_mlflow(best_model,classification_train_metric)


            y_test_pred=best_model.predict(x_test)
            classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

            self.track_mlflow(best_model,classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            Churn_Model=ChurnModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=Churn_Model)
            #model pusher
            save_object("final_model/model.pkl",best_model)
            

            ## Model Trainer Artifact
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                train_metric_artifact=classification_train_metric,
                                test_metric_artifact=classification_test_metric
                                )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise CustomerChurnException(e,sys)