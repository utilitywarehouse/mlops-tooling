from google.cloud import aiplatform
from mlops_tooling.logger.main import get_logger

logger = get_logger(__name__)

def create_batch_run(
    project: str,
    display_name: str,
    model_id: str,
    instances_format: str,
    gcs_source_uri: str,
    predictions_format: str,
    gcs_destination_output_uri_prefix: str,
    machine_type: str =  "n1-standard-2",
    accelerator_type: str = None,
    accelerator_count: int = 0,
    location: str = "europe-west2",
    api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)

    batch_prediction_job = {
        "display_name": display_name,
        "model": f'projects/{project}/locations/{location}/models/{model_id}',
        "input_config": {
            "instances_format": instances_format,
            "gcs_source": {"uris": [gcs_source_uri]},
        },
        "output_config": {
            "predictions_format": predictions_format,
            "gcs_destination": {"output_uri_prefix": gcs_destination_output_uri_prefix},
        },
        "dedicated_resources": {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "starting_replica_count": 1,
            "max_replica_count": 1,
        },
    }
    
    parent = f"projects/{project}/locations/{location}"
    
    response = client.create_batch_prediction_job(
        parent=parent, batch_prediction_job=batch_prediction_job
    )
    logger.info(response)
