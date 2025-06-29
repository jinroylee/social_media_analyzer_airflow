#!/bin/bash
pip install mlflow==2.12.1 gunicorn
export MLFLOW_BACKEND_STORE_URI="file:///mlflow/mlruns"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="file:///mlflow/mlruns"
gunicorn -c /gunicorn_config.py mlflow.server:app 