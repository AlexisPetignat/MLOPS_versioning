import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri(uri="http://localhost:8080")

filepath = "https://github.com/ybifoundation/Dataset/raw/main/Fish.csv"

data = pd.read_csv(filepath)

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

params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42,
}


def train(params=params):
    model = GradientBoostingRegressor(**(params or {}))

    # Train
    model.fit(X_train, y_train)
    return model


model = train()
y_pred = model.predict(X_test)
metrics = {"r2": r2_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred)}

# Logging
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("r2 Score", metrics["r2"])
    mlflow.log_metric("MSE", metrics["mse"])

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="Les espaces verts dans paris",
        signature=signature,
        input_example=X_train,
        registered_model_name="Le arrondissement model_id",
    )

    # Set a tag that we can use to remind ourselves what this model was for
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Lr for le espaces verts"}
    )
