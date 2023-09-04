from google.cloud import aiplatform
from mlops_tooling.logger.main import get_logger

logger = get_logger(__name__)

## NOTE: This area contains two nearly identical functions.
## Both functions will end up with the same results, however the means by which they achieve their goals are different.
## The first of the two functions, create_batch_run, starts a batch job in the middle of the script, waits for the job to finish, then continues.
## The second of the two functions, create_batch_job, schedules a batch job on the cloud at the moment it is ran, and then continues while the job runs in the background.
## If you just want to create a job, use the latter. If you want to create a job and then do something with the output, use the former.

def create_batch_run(
    project: str,
    display_name: str,
    model_id: str,
    instances_format: str,
    gcs_source_uri: str,
    predictions_format: str,
    gcs_destination_output_uri_prefix: str,
    batch_size: int = 64,
    machine_type: str =  "n1-standard-2",
    accelerator_type: str = None,
    accelerator_count: int = 0,
    location: str = "europe-west2",
    api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
):
    # NB: This function runs as a normal function would.
    # It will run until the job has completed, and then you can continue with your workflow.
    
    aiplatform.init(project=project, location=location)
    model = aiplatform.Model(f'projects/{project}/locations/{location}/models/{model_id}')

    if accelerator_type:
        batch_prediction_job = model.batch_predict(
            job_display_name = display_name,
            gcs_source = gcs_source_uri,
            instances_format = instances_format,
            gcs_destination_prefix = gcs_destination_output_uri_prefix,
            predictions_format = predictions_format,
            machine_type = machine_type,
            accelerator_type = accelerator_type,
            accelerator_count = accelerator_count,
            batch_size = batch_size
        )
        
    else:
        batch_prediction_job = model.batch_predict(
            job_display_name = display_name,
            gcs_source = gcs_source_uri,
            instances_format = instances_format,
            gcs_destination_prefix = gcs_destination_output_uri_prefix,
            predictions_format = predictions_format,
            machine_type = machine_type,
            batch_size = batch_size
        )

    return batch_prediction_job

def create_batch_job(
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
    # NB: This function works as an async function
    # We call it and it runs in the background, allowing us to continue working.
    
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
