import argparse
import glob
import os

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUMERIC_COLS = [
    'Pregnancies',
    'PlasmaGlucose',
    'DiastolicBloodPressure',
    'TricepsThickness',
    'SerumInsulin',
    'BMI',
    'DiabetesPedigree',
    'Age']

TARGET_COL = 'Diabetic'


def log_azure_ml():
    # "MLFLOW_TRACKING_URI" is set-up when running inside an Azure ML Job
    # This command only needs to run once
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Tags are shown as properties of the job in
    # the Azure ML dashboard. Run once.
    # tags = {"tag": "example"}
    # mlflow.set_tags(tags)

    # If you log metrics with same key multiple times,
    # you get a plot in Azure ML
    # metrics = {"answer": 42}
    # mlflow.log_metrics(metrics)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(data_path):
    data = get_csvs_df(data_path)
    data = data[NUMERIC_COLS + [TARGET_COL]]

    # split data
    random_data = np.random.rand(len(data))

    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85

    train = data[msk_train]
    val = data[msk_val]
    test = data[msk_test]

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('val size', val.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    return train, val, test


def train(train_data: pd.DataFrame, reg_rate: float):
    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS]

    # train model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear") \
        .fit(X_train, y_train)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("reg_rate", reg_rate)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)

    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train, color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    return model


def run_script(data_path: str, reg_rate: float, logging_enabled=True):
    if logging_enabled:
        log_azure_ml()

    train_data, val_data, test_data = split_data(data_path)
    train(train_data, reg_rate)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        help="Path where data is stored")
    parser.add_argument("--reg_rate",
                        default=0.01,
                        required=False,
                        type=float,
                        help="Regularization rate")
    args = parser.parse_args()
    run_script(args.data_path, args.reg_rate)
