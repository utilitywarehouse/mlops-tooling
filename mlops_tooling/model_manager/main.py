from google.cloud import aiplatform
from pathlib import Path
import re
import os
import mlflow
from mlflow import MlflowClient


class ModelManager:
    def __init__(
        self,
        tracking_uri: str,
        google_project: str,
        gcs_bucket: str,
        google_credentials: str = None,
        api_location: str = "europe-west2",
        api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
        serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    ):
        self.tracking_uri = tracking_uri
        self.google_project = google_project
        self.gcs_bucket = gcs_bucket
        self.api_endpoint = api_endpoint
        self.api_location = api_location
        self.serving_container_image_uri = serving_container_image_uri

        self.set_google_credentials(google_credentials)
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_client = MlflowClient()

        aiplatform.init(project=self.google_project, location=self.api_location)

    @staticmethod
    def set_google_credentials(google_credentials: str = None):
        if not google_credentials:
            google_credentials = (
                re.findall("^(\/[^\/]*\/[^\/]*\/)\/?", str(Path().absolute()))[0]
                + ".config/gcloud/dev.json"
            )

        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        except:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

    def model_info(self, model_name: str):
        model_info = self.mlflow_client.get_registered_model(model_name)
        return model_info

    def model_uri(self, model_name: str, model_stage: str = "Production"):
        model_info = self.model_info(model_name)
        model_versions = model_info.latest_versions

        uri_snippet = re.search(
            ".+:(.+)",
            list(filter(lambda x: x.current_stage == model_stage, model_versions))[
                0
            ].source,
        )[1]

        model_uri = self.gcs_bucket + "/mlflow" + uri_snippet
        return model_uri

    def upload_model(
        self,
        model_name: str,
        model_description: str,
        model_display_name: str = None,
        model_stage: str = "Production",
    ):
        model_display_name = model_display_name if model_display_name else model_name

        model_uri = self.model_uri(model_name=model_name, model_stage=model_stage)

        model = aiplatform.Model.upload(
            display_name=model_display_name,
            description=model_description,
            artifact_uri=model_uri,
            serving_container_image_uri=self.serving_container_image_uri,
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
        endpoint = aiplatform.Endpoint.create(
            display_name=model_name,
            project=self.google_project,
            location=self.api_location,
        )

        endpoint_id = re.search(".+/endpoints/(\d+)$", endpoint.resource_name)[1]

        return endpoint_id

    def deploy_model(self, model_url, endpoint_id):
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
        model_stage: str = "Production",
    ):
        model_url = self.upload_model(
            model_name, model_description, model_display_name, model_stage
        )
        endpoint_id = self.create_model_endpoint(model_name)

        self.deploy_model(model_url, endpoint_id)

    def create_new_experiment(self, experiment_name):
        mlflow.create_experiment(experiment_name)

    def log_results(
        self,
        experiment_name,
        run_name,
        model,
        parameters: dict,
        metrics: dict,
        model_type: str = "sklearn",
    ):
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name = run_name):
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            eval(f"mlflow.{model_type}.log_model(model, 'model')")
