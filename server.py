from train import train
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Model default parameters
params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42,
}

global model
model = train(params)
app = FastAPI()


class ModelParams(BaseModel):
    n_estimators: int
    learning_rate: float
    max_depth: int
    random_state: int


@app.post("/update-model")
async def updateModel(parameters: ModelParams):
    params = parameters.model_dump()
    global model
    model = train(params)
    return params


class fish(BaseModel):
    Height: float
    Width: float
    Length1: float
    Length2: float
    Length3: float


@app.post("/predict")
async def predict(le_fish: fish):
    global model
    model = train(params)
    return model.predict(pd.DataFrame([le_fish.model_dump()])).tolist()
