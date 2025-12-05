import glob
import os
import re
import uuid
from pathlib import Path

import mlflow
from google.cloud import storage
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from mlops_tooling.logger.main import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(
        self,
        tracking_uri: str,
        google_project: str,
        gcs_bucket: str,
        google_credentials: str = None,
        api_location: str = "europe-west2",
    ):
        self.tracking_uri = tracking_uri
        self.google_project = google_project
        self.gcs_bucket = gcs_bucket
        self.api_location = api_location

        if google_credentials:
            self.google_credentials = google_credentials
        else:
            self.google_credentials = self.set_google_credentials(self.google_project)

        self.set_google_credentials(self.google_credentials)
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_client = MlflowClient()

    @staticmethod
    def set_google_credentials(project):
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            return os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        else:
            if "dev" in project:
                proj = "dev"
            else:
                proj = "prod"

            google_credentials = (
                re.findall("^(\/[^\/]*\/[^\/]*\/)\/?", str(Path().absolute()))[0]
                + f".config/gcloud/{proj}.json"
            )

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

            return google_credentials

    def model_info(self, model_name: str):
        """
        Pulls information about a model from MLflow.

        Parameters
        ----------
        model_name : str
            The name of the model to pull information about.

        Returns
        ----------
        model_info : dict
            A dictionary containing information about the model.
        """
        model_info = self.mlflow_client.get_registered_model(model_name)
        return model_info

    def model_location(self, model_type: str):
        if model_type == "tensorflow":
            return "/data/model"
        else:
            return ""

    def model_uri(
        self,
        model_name: str,
        model_type: str = "sklearn",
        model_stage: str = "Production",
    ):
        """
        Finds the Google Cloud URI of a model in MLflow.

        Parameters
        ----------
        model_name : str
            The name of the model to return the URI for.
        model_stage : str
            The stage of the model to return the URI for.

        Returns
        ----------
        model_uri : str
            The Google Cloud Storage URI for the model.
        """
        model_info = self.model_info(model_name)
        model_versions = model_info.latest_versions

        uri_snippet = re.search(
            ".+:(.+)",
            list(filter(lambda x: x.current_stage == model_stage, model_versions))[
                0
            ].source,
        )[1]

        model_location = self.model_location(model_type=model_type)
        model_uri = self.gcs_bucket + "/mlflow" + uri_snippet + model_location
        return model_uri

    def create_new_experiment(self, experiment_name):
        """
        Creates a new experiment in MLflow.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to create in MLflow.
        """
        mlflow.create_experiment(experiment_name)

    @staticmethod
    def get_signature(model, X):
        signature = infer_signature(X, model.predict(X))

        return signature

    def log_results(
        self,
        experiment_name,
        run_name,
        model,
        signature=None,
        parameters: dict = None,
        metrics: dict = None,
        artifacts: str = None,
        model_type: str = "sklearn",
        override_model_artifacts: bool = False,
        model_artifact_folder: str = None,
    ):
        """
        Logs a model to MLflow.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment to log results to.
        run_name : str
            Name of the specific run to log.
        model : sklearn-style model
            The model file to log to MLflow.
        parameters : dict
            A dictionary of parameters used to create the model, which will be saved to MLflow.
        metrics : dict
            A dictionary of model metrics, which will be saved to MLflow.
        model_type : str, default "sklearn"
            The type of the model to save, usually sklearn, however this can be lightgbm, xgboost, etc.
        """
        mlflow.set_experiment(experiment_name)
        # artifact_uri = mlflow.get_artifact_uri()
        # mlflow.get_
        # logger.info(artifact_uri)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("mlflow.runName", run_name)

            if parameters:
                mlflow.log_params(parameters)

            if metrics:
                mlflow.log_metrics(metrics)

            if override_model_artifacts:
                gcs_uri = mlflow.get_artifact_uri()

                if not model_artifact_folder:
                    sha = str(uuid.uuid4().hex)
                    ## this dumps our model to a model folder
                    model_artifact_folder = f"mlflow-models/{sha}"
                    logger.info(
                        f"Saving model to the following folder: {model_artifact_folder}"
                    )
                    eval(
                        f"mlflow.{model_type}.save_model(model, '{model_artifact_folder}')"
                    )

                mlflow.end_run()

                ## The above returns a URI like so: 'gs://bucket/mlflow/experiment_id/run_id/artifacts'
                ## We remove the gs://bucket from it and add in /run_name/model/data to the end to ensure our folder ends up in the correct place.
                gcs_uri = "mlflow/" + "/".join(gcs_uri.split(":/")[1:]) + "/" + run_name

                logger.info(f"Overriding to the following uri: {gcs_uri}")

                self.override_upload_artifacts(
                    self.google_credentials,
                    self.google_project,
                    self.gcs_bucket.split("//")[1],
                    gcs_uri,
                    model_artifact_folder,
                )

                logger.info("Model uploaded.")

            else:
                if artifacts:
                    mlflow.log_artifacts(artifacts)

                eval(f"mlflow.{model_type}.log_model(model, '{run_name}')")

        return run.info.run_id

    def override_upload_artifacts(
        self,
        service_account: str,
        project: str,
        bucket_name: str,
        gcs_path: str,
        local_path: str,
    ):
        """
        Uploads a local folder to GCS.

        Args:
            service_account (str): Location of your SA.
            project (str): Name of the project where you GCS bucket is, e.g. "uw-example-project"
            bucket_name (str): Name of the bucket your want to upload to, e.g. "uw-example-bucket"
            gcs_path (str): Name of the location in the bucket you want to save to , e.g. "model-1"
            local_path (str): Folder name to upload.
        """
        client = storage.Client.from_service_account_json(
            json_credentials_path=service_account, project=project
        )
        bucket = client.bucket(bucket_name)

        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + "/**"):
            if not os.path.isfile(local_file):
                logger.info(f"Found folder {local_file}, writing to to {gcs_path}")
                self.override_upload_artifacts(
                    service_account,
                    project,
                    bucket_name,
                    gcs_path + "/" + os.path.basename(local_file),
                    local_file,
                )
            else:
                filename = local_file.split("/")[-1]
                blob = bucket.blob(gcs_path + "/" + filename)
                blob.upload_from_filename(local_file, timeout=600)
                logger.info(f"Wrote object to {gcs_path}")
