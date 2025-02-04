import os
import json
import mlflow
import argparse
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import asdict
from mlflow.tracking import MlflowClient
from config.serve_config import BaseServeArgs
from utils import Logger, AppPath

load_dotenv()
LOGGER = Logger(name=__file__, 
                log_file="model_registry.log")
LOGGER.log.info("Start model registry...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str,
                        help="Name of the config file")
    parser.add_argument("--filter_string", type=str, default="",
                        help="Filter string for searching runs in MLflow tracking server")
    parser.add_argument("--best_metric", type=str, choices=["best_val_loss", "best_val_acc"], default="best_val_loss",
                        help="Metric for selecting the best model")
    parser.add_argument("--model_alias", type=str, default="Production", 
                        help="Alias tag of the model. Help to identify the model in the model registry.")

    args = parser.parse_args()

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    LOGGER.log.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    experiment_ids = dict(mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME))['experiment_id']
    
    client = MlflowClient()

    try:
        # search experiments 
        best_runs = client.search_runs(experiment_ids, 
                                       filter_string=args.filter_string,
                                       order_by=[f"metrics.{args.best_metric} DESC"]
                                       )[-1]

    except Exception as e:
        LOGGER.log.error("Error occurred while registy model: " + str(e))
        LOGGER.log.info("Runs not found")
        exit(0)
    
    model_name = best_runs.data.params['model_name']

    try:
        # registry model
        client.create_registered_model(name=model_name)
    except:
        pass

    # Perform the steps to register and assign an alias to a model in MLflow
    run_id = best_runs.info.run_id # Get the run_id of the best model
    model_uri = f"runs:/{run_id}/model" # create model URI
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id) # Register a new version of the model to the Model Registry
    LOGGER.log.info(f"Registered model: {model_name}, version: {mv.version}")
    client.set_registered_model_alias(name=model_name, alias=args.model_alias, version=mv.version)

    server_config = BaseServeArgs(config_name=args.config_name, 
                                  model_name=model_name,
                                  model_alias=args.model_alias)
    
    path_save_cfg = AppPath.ARTIFACTS / f"{args.config_name}.json"
    with open(path_save_cfg, mode="w+") as f:
        json.dump(obj=asdict(server_config), fp=f, indent=4)
    
    LOGGER.log.info(f"Config saved to {args.config_name}.json")

    LOGGER.log.info(f"Model {model_name} registered with alias {args.model_alias} and version {mv.version}")
    LOGGER.log.info("Model Registry completed")

if __name__ == "__main__":
    main()