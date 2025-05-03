from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading model pipeline...")
    #app.state.model_pipeline = joblib.load("../reddit_model_pipeline.joblib")
    app.state.model_pipeline = joblib.load("reddit_model_pipeline.joblib")
    print("model loaded successfully!")
    yield
    print("server shutting down... Bye!")

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
    lifespan=lifespan
)

class request_body(BaseModel):
    reddit_comment : str

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying Reddit comments'}

# class request_body(BaseModel):
#     reddit_comment : str

# @app.on_event('startup')
# def load_artifacts():
#     global model_pipeline
#     model_pipeline = joblib.load("../reddit_model_pipeline.joblib")


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    X = [data.reddit_comment]
    model_pipeline = app.state.model_pipeline
    predictions = model_pipeline.predict_proba(X)
    return {'Predictions': predictions.tolist()}

# Defining path operation for /name endpoint
# @app.get('/{name}')
# def hello_name(name : str):
# 	return {'message': f'Hello {name}'}
