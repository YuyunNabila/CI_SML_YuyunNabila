import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
# CONFIG MLFLOW

mlflow.set_experiment("Student Performance Modelling")

# LOAD DATA
df = pd.read_csv("student_performance_clean.csv")

target = "math score"

X = df.drop(columns=[target])
y = df[target]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL
with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # LOGGING

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(model, "model")

    print("Training selesai & logged ke MLflow ")
