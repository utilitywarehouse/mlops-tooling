# MLOps Tooling

A package for UtilityWarehouse's internal machine learning tooling. This package contains three subpackages, bq_connector, model_manager, and Timeseries.

## BigQuery connector

BigQuery connector uses Jinja2 to enable the user to write .sql files and import them into their notebooks.

## Model Manager

Model manager is a tool used to link log models to MLflow and deploy them on Vertex AI.

## Timeseries

Timeseries is a class that speeds up the created of flattened timeseries datasets for use with tree-based models. The package also contains a 'plot forecast' function, that can be used to easily plot a timeseries.
