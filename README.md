# MLOps Tooling

A package for UtilityWarehouse's internal machine learning tooling. This package contains two subpackages, model_manager and bq_connector.

## Model Manager

Model manager is a tool used to link log models to MLflow and deploy them on Vertex AI. We can use it like so:

```python
from modelmanager import ModelManager

TRACKING_URI = "https://mlflow.host.site"
GOOGLE_PROJECT = 'data-project'
GCS_BUCKET =  "gs://data-bucket"
LOCATION = "europe-west2"
API_ENDPOINT = "europe-west2-aiplatform.googleapis.com"
SCORING_METRIC = "sklearn_metric_name"

model = your_model()

grid_search_parameters = {
  paramters_here
}

grid_search = GridSearchCV(
  estimator = model,
  param_grid = grid_search_parameters,
  scoring = SCORING_METRIC,
  cv=5,
  n_jobs=-1,
  verbose=3
)

grid_search.fit(train_X, train_y)

best_parameters = grid_search.best_params_
best_metrics = {“score” : grid_search.best_score_} 

mlflow = ModelManager(
    tracking_uri=TRACKING_URI,
    google_project=GOOGLE_PROJECT,
    gcs_bucket=GCS_BUCKET
)

model_name = "Model name"
model_display_name = "Display name"
experiment_name = model_name + " experiment"

mlflow.create_new_experiment(experiment_name)

mlflow.log_results(
    experiment_name=experiment_name,
    model=optimal_model,
    parameters=best_parameters,
    metrics=best_metrics
)


mlflow = ModelManager(
    tracking_uri=TRACKING_URI,
    google_project=GOOGLE_PROJECT,
    gcs_bucket=GCS_BUCKET
)

mlflow.serve_model(
    model_name=model_name,
    model_description="Model description",
    model_display_name=model_display_name
)
```

## BigQuery connector

BigQuery connector uses Jinja2 to enable the user to write .sql files and import them into their notebooks. We can use by

```python
bq = BigQuery(path = "path_to_sql_files", credentials = "ceredentials/path")
data = bq.query("example_query.sql")
```
# mlops-tooling
