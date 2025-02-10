import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(f"MongoDB URL: {mongo_db_url}")

import pymongo
from customer_churn.Exception.exception import CustomerChurnException
from customer_churn.Logging.logger import logging
from customer_churn.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from customer_churn.utils.main_utils.utils import load_object
from customer_churn.utils.ml_utils.model.estimator import ChurnModel

# Connect to MongoDB
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
from customer_churn.Constants.training_pipeline import (DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Initialize FastAPI app
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise CustomerChurnException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Load preprocessor and model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        # Initialize model
        network_model = ChurnModel(preprocessor=preprocessor, model=final_model)

        # Make predictions
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        # Ensure 'prediction_output' directory exists
        output_dir = "prediction_output"
        os.makedirs(output_dir, exist_ok=True)

        # Save the output file
        output_path = os.path.join(output_dir, "output.csv")
        df.to_csv(output_path, index=False)

        # Convert dataframe to HTML table
        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise CustomerChurnException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
