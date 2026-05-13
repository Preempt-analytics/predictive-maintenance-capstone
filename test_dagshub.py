import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with mlflow.start_run():
    mlflow.log_param("test", "connection_check")
    mlflow.log_metric("dummy_metric", 1.0)

print("Connection successful")
