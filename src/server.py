from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np


# Model default parameters
params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42,
}


filepath = "https://github.com/ybifoundation/Dataset/raw/main/Fish.csv"


def load():
    # Load data
    data = pd.read_csv(filepath)
    print(data.columns)
    X = data[["Height", "Width", "Length1", "Length2", "Length3"]]
    y = data["Weight"]
    test_size = 0.4

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train(params=params):
    X_train, X_test, y_train, y_test = load()
    model = GradientBoostingRegressor(**(params or {}))

    # Train
    model.fit(X_train, y_train)
    return model


# Params
current = train(params)
next = train(params)
p = 0.5
app = FastAPI()


class fish(BaseModel):
    Height: float
    Width: float
    Length1: float
    Length2: float
    Length3: float


@app.post("/predict")
async def predict(le_fish: fish):
    global current, next

    if np.random.rand() < p:
        return current.predict(pd.DataFrame([le_fish.model_dump()])).tolist()
    return next.predict(pd.DataFrame([le_fish.model_dump()])).tolist()


class ModelParams(BaseModel):
    n_estimators: int
    learning_rate: float
    max_depth: int
    random_state: int


@app.post("/update-model")
async def updateModel(parameters: ModelParams):
    params = parameters.model_dump()
    global next
    next = train(params)
    return params


@app.post("/accept-next-model")
async def acceptNextModel():
    global current, next
    current = next
    return {"status": "accepted"}


@app.post("/update-p")
async def updateP(new_p: float):
    global p
    p = new_p
    return {"p": p}
