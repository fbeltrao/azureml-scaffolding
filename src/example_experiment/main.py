import argparse
import os

import mlflow

from common.printer import print_data_path

def greet_world(greeting: str):
    print(f"{greeting} world!")

def log_azure_ml():
    # "MLFLOW_TRACKING_URI" is set-up when running inside an Azure ML Job
    # This command only needs to run once
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Tags are shown as properties of the job in the Azure ML dashboard. Run once.
    tags = {"tag": "example"}
    mlflow.set_tags(tags)

    # If you log metrics with same key multiple times, you get a plot in Azure ML
    metrics = {"answer": 42}
    mlflow.log_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path where data is stored")
    parser.add_argument("--greeting", type=str, help="Word with which to greet the world")
    args = parser.parse_args()

    greet_world(greeting=args.greeting)
    print_data_path(args.data_path)
    log_azure_ml()
