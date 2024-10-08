import glob
import os
import re
import uuid
from pathlib import Path

import mlflow
from google.cloud import aiplatform, storage
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
        api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
    ):
        self.tracking_uri = tracking_uri
        self.google_project = google_project
        self.gcs_bucket = gcs_bucket
        self.api_endpoint = api_endpoint
        self.api_location = api_location

        if google_credentials:
            self.google_credentials = google_credentials
        else:
            self.google_credentials = self.set_google_credentials(self.google_project)

        self.set_google_credentials(self.google_credentials)
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_client = MlflowClient()

        aiplatform.init(project=self.google_project, location=self.api_location)

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

    def upload_model(
        self,
        model_name: str,
        model_description: str,
        model_display_name: str = None,
        model_type: str = "sklearn",
        model_stage: str = "Production",
        serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    ):
        """
        Uploads an MLflow model to a Vertex AI, allowing it to be attached to an endpoint.

        Parameters
        ----------
        model_name : str
            The name of the model in MLflow to upload to Vertex AI.
        model_description : str
            A description of the model, which will be displayed in Vertex AI.
        model_display_name : str, optional
            Chosen name of the model, which will be displayed in Vertex AI. This defaults to the model name.
        model_stage : str
            The stage of the model to upload. Can be production or staging.

        Returns
        ----------
        model_url : str
            The Vertex AI url for the model.
        """
        assert model_stage in ["Staging", "Production"]
        model_display_name = model_display_name if model_display_name else model_name

        model_uri = self.model_uri(
            model_name=model_name, model_stage=model_stage, model_type=model_type
        )

        model = aiplatform.Model.upload(
            display_name=model_display_name,
            description=model_description,
            artifact_uri=model_uri,
            serving_container_image_uri=serving_container_image_uri,
            sync=True,
        )

        model.wait()

        model_vertex_id = model.name
        model_url = (
            "projects/"
            + self.google_project
            + "/locations/europe-west2/models/"
            + model_vertex_id
        )

        return model_url

    def create_model_endpoint(self, model_name: str):
        """
        Creates a Vertex AI endpoint for a model in MLflow.

        Parameters
        ----------
        model_name : str
            The name of the model in MLflow to upload to Vertex AI, which will be used to name the endpoint.

        Returns
        ----------
        endpoint_id : str
            The Vertex AI endpoint ID for the model.
        """
        endpoint = aiplatform.Endpoint.create(
            display_name=model_name,
            project=self.google_project,
            location=self.api_location,
        )

        endpoint_id = re.search(".+/endpoints/(\d+)$", endpoint.resource_name)[1]

        return endpoint_id

    def deploy_model(self, model_url: str, endpoint_id: str):
        """
        Deploys a model in Vertex AI to a Vertex AI endpoint.

        Parameters
        ----------
        model_url : str
            The url of the model uploaded to Vertex AI.
        endpoint_id : str
            The Vertex AI endpoint ID for the model to be deployed to.
        """
        client_options = {"api_endpoint": self.api_endpoint}

        client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)

        deployed_model = {
            "model": model_url,
            "dedicated_resources": {
                "min_replica_count": 1,
                "machine_spec": {
                    "machine_type": "n1-standard-2",
                },
            },
        }

        traffic_split = {"0": 100}

        endpoint = client.endpoint_path(
            project=self.google_project,
            location=self.api_location,
            endpoint=endpoint_id,
        )

        response = client.deploy_model(
            endpoint=endpoint,
            deployed_model=deployed_model,
            traffic_split=traffic_split,
        )

    def serve_model(
        self,
        model_name: str,
        model_description: str,
        model_display_name: str = None,
        model_type: str = "sklearn",
        model_stage: str = "Production",
        serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    ):
        """
        Uploads an MLflow model to a Vertex AI, creates an endpoint for the model,
        and deploys the model to the endpoint.

        Parameters
        ----------
        model_name : str
            The name of the model in MLflow to upload to Vertex AI.
        model_description : str
            A description of the model, which will be displayed in Vertex AI.
        model_display_name : str, optional
            Chosen name of the model, which will be displayed in Vertex AI. This defaults to the model name.
        model_stage : str
            The stage of the model to upload. Can be production or staging.
        """
        model_url = self.upload_model(
            model_name,
            model_description,
            model_display_name,
            model_type,
            model_stage,
            serving_container_image_uri,
        )
        endpoint_id = self.create_model_endpoint(model_name)

        self.deploy_model(model_url, endpoint_id)

    def create_new_experiment(self, experiment_name):
        """
        Creates a new experiment in MLflow.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to create in MLflow.
        """
        mlflow.create_experiment(experiment_name)

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


class DualModelManager:
    def __init__(
        self,
        dev_tracking_uri: str,
        prod_tracking_uri: str,
        dev_google_project: str,
        prod_google_project: str,
        dev_gcs_bucket: str,
        prod_gcs_bucket: str,
        google_credentials: str = None,
        api_location: str = "europe-west2",
        api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
    ):
        self.dev_tracking_uri = dev_tracking_uri
        self.prod_tracking_uri = prod_tracking_uri

        self.dev_google_project = dev_google_project
        self.prod_google_project = prod_google_project

        self.dev_gcs_bucket = dev_gcs_bucket
        self.prod_gcs_bucket = prod_gcs_bucket

        self.api_endpoint = api_endpoint
        self.api_location = api_location

        self.google_credentials = google_credentials

        self.set_google_credentials(self.google_credentials)

        self.dev_mlflow_client = MlflowClient(uri=self.dev_tracking_uri)
        self.prod_mlflow_client = MlflowClient(uri=self.prod_tracking_uri)

    @staticmethod
    def _set_ai_platform(google_project, api_location):
        aiplatform.init(project=google_project, location=api_location)

    def set_ai_platform(self, environment: str = "dev"):
        api_location = self.api_location

        if environment == "dev":
            google_project = self.dev_google_project
        else:
            google_project = self.prod_google_project

        self._set_ai_platform(google_project, api_location)

    def set_google_credentials(self, project):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_credentials

        return self.google_credentials

    def _mlflow_client(self, environment: str = "dev"):
        if environment == "dev":
            return self.dev_mlflow_client
        else:
            return self.prod_mlflow_client

    def model_info(self, model_name: str, environment: str = "dev"):
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
        client = self._mlflow_client(environment)
        model_info = client.get_registered_model(model_name)

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
        environment: str = "dev",
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
        model_info = self.model_info(model_name, environment)
        model_versions = model_info.latest_versions

        uri_snippet = re.search(
            ".+:(.+)",
            list(filter(lambda x: x.current_stage == model_stage, model_versions))[
                0
            ].source,
        )[1]

        model_location = self.model_location(model_type=model_type)

        if environment == "dev":
            gcs_bucket = self.dev_gcs_bucket
        else:
            gcs_bucket = self.prod_gcs_bucket

        model_uri = gcs_bucket + "/mlflow" + uri_snippet + model_location

        return model_uri

    def upload_model(
        self,
        model_name: str,
        model_description: str,
        model_display_name: str = None,
        model_type: str = "sklearn",
        model_stage: str = "Production",
        environment: str = "dev",
        serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    ):
        """
        Uploads an MLflow model to a Vertex AI, allowing it to be attached to an endpoint.

        Parameters
        ----------
        model_name : str
            The name of the model in MLflow to upload to Vertex AI.
        model_description : str
            A description of the model, which will be displayed in Vertex AI.
        model_display_name : str, optional
            Chosen name of the model, which will be displayed in Vertex AI. This defaults to the model name.
        model_stage : str
            The stage of the model to upload. Can be production or staging.

        Returns
        ----------
        model_url : str
            The Vertex AI url for the model.
        """
        assert model_stage in ["Staging", "Production"]
        model_display_name = model_display_name if model_display_name else model_name

        model_uri = self.model_uri(
            model_name=model_name,
            model_stage=model_stage,
            model_type=model_type,
            environment=environment,
        )

        self.set_ai_platform(environment)

        model = aiplatform.Model.upload(
            display_name=model_display_name,
            description=model_description,
            artifact_uri=model_uri,
            serving_container_image_uri=serving_container_image_uri,
            sync=True,
        )

        model.wait()

        if environment == "dev":
            google_project = self.dev_google_project
        else:
            google_project = self.prod_google_project

        model_vertex_id = model.name
        model_url = (
            "projects/"
            + google_project
            + "/locations/europe-west2/models/"
            + model_vertex_id
        )

        return model_url

    def create_model_endpoint(self, model_name: str, environment: str = "dev"):
        """
        Creates a Vertex AI endpoint for a model in MLflow.

        Parameters
        ----------
        model_name : str
            The name of the model in MLflow to upload to Vertex AI, which will be used to name the endpoint.

        Returns
        ----------
        endpoint_id : str
            The Vertex AI endpoint ID for the model.
        """
        self.set_ai_platform(environment)

        endpoint = aiplatform.Endpoint.create(
            display_name=model_name,
            project=self.google_project,
            location=self.api_location,
        )

        endpoint_id = re.search(".+/endpoints/(\d+)$", endpoint.resource_name)[1]

        return endpoint_id

    def deploy_model(self, model_url: str, endpoint_id: str, environment: str = "dev"):
        """
        Deploys a model in Vertex AI to a Vertex AI endpoint.

        Parameters
        ----------
        model_url : str
            The url of the model uploaded to Vertex AI.
        endpoint_id : str
            The Vertex AI endpoint ID for the model to be deployed to.
        """
        self.set_ai_platform(environment)

        client_options = {"api_endpoint": self.api_endpoint}

        client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)

        deployed_model = {
            "model": model_url,
            "dedicated_resources": {
                "min_replica_count": 1,
                "machine_spec": {
                    "machine_type": "n1-standard-2",
                },
            },
        }

        traffic_split = {"0": 100}

        if environment == "dev":
            google_project = self.dev_google_project
        else:
            google_project = self.prod_google_project

        endpoint = client.endpoint_path(
            project=google_project,
            location=self.api_location,
            endpoint=endpoint_id,
        )

        response = client.deploy_model(
            endpoint=endpoint,
            deployed_model=deployed_model,
            traffic_split=traffic_split,
        )

    def serve_model(
        self,
        model_name: str,
        model_description: str,
        model_display_name: str = None,
        model_type: str = "sklearn",
        model_stage: str = "Production",
        environment: str = "dev",
        serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    ):
        """
        Uploads an MLflow model to a Vertex AI, creates an endpoint for the model,
        and deploys the model to the endpoint.

        Parameters
        ----------
        model_name : str
            The name of the model in MLflow to upload to Vertex AI.
        model_description : str
            A description of the model, which will be displayed in Vertex AI.
        model_display_name : str, optional
            Chosen name of the model, which will be displayed in Vertex AI. This defaults to the model name.
        model_stage : str
            The stage of the model to upload. Can be production or staging.
        """
        model_url = self.upload_model(
            model_name,
            model_description,
            model_display_name,
            model_type,
            model_stage,
            environment,
            serving_container_image_uri,
        )
        endpoint_id = self.create_model_endpoint(model_name, environment)

        self.deploy_model(model_url, endpoint_id, environment)

    def create_new_experiment(self, experiment_name, environment: str = "dev"):
        """
        Creates a new experiment in MLflow.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to create in MLflow.
        """
        client = self._mlflow_client(environment)
        client.create_experiment(experiment_name)

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
        environment: str = "dev",
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
        if environment == "dev":
            uri = self.dev_tracking_uri
            google_project = self.dev_google_project
            gcs_bucket = self.dev_gcs_bucket
        else:
            uri = self.prod_tracking_uri
            google_project = self.prod_google_project
            gcs_bucket = self.prod_gcs_bucket

        mlflow.set_tracking_uri(uri)

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
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
                    google_project,
                    gcs_bucket.split("//")[1],
                    gcs_uri,
                    model_artifact_folder,
                )

                logger.info("Model uploaded.")

            else:
                if artifacts:
                    mlflow.log_artifacts(artifacts)

                eval(f"mlflow.{model_type}.log_model(model, '{run_name}')")

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
