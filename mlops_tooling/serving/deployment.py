import subprocess
import re
from mlops_tooling.logger import get_logger

logger = get_logger(__name__)


def run_subprocess(command, capture_output=False):
    subprocess.run(command, shell=True, check=True, capture_output=capture_output)


def run_command(command, capture_output=False):
    try:
        run_subprocess(command, capture_output)

    except Exception as e:
        logger.error(f"Error: {e}")


def set_project(project):
    command = f"gcloud config set project {project}"

    run_command(command)


def create_artifact_repository(repository_name, region, description):
    command = f"gcloud artifacts repositories create {repository_name} --repository-format=docker --location='{region}' --description='{description}'"
    run_command(command)


def create_build(project, repository, region, image):
    command = f"gcloud builds submit --region='{region}' --tag='{region}-docker.pkg.dev/{project}/{repository}/{image}'"
    run_command(command)


def create_model(project, repository, region, image, model_name, env_vars=None):
    command = f"gcloud ai models upload --container-ports=80 --container-predict-route='/predict' --container-health-route='/health' --region='{region}' --display-name='{model_name}' --container-image-uri='{region}-docker.pkg.dev/{project}/{repository}/{image}'"

    if env_vars:
        add_ons = ",".join(env_vars)
        command += add_ons

    run_command(command)


def create_model_endpoint(project, region, model_name):
    command = f"gcloud ai endpoints create --project={project} --region={region} --display-name={model_name}"

    run_command(command)


def find_model(project, region, model_name):
    command = f"gcloud ai models list --project='{project}' --region='{region}' --filter='displayName={model_name}'"

    model = run_command(command, capture_output=True)

    model_id = re.search("(\d+)", model.stdout.decode("utf-8"))

    return model_id


def find_endpoint(project, region, endpoint_name):
    command = f"gcloud ai endpoints list --project='{project}' --region='{region}' --filter='displayName={endpoint_name}'"

    endpoint = run_command(command, capture_output=True)

    endpoint_id = re.search("(\d+)", endpoint.stdout.decode("utf-8"))

    return endpoint_id


def deploy_model(
    project, region, model_name, endpoint_id, model_id, machine_type="n1_standard-2"
):
    command = f"gcloud ai endpoints deploy-model {endpoint_id} --project='{project}' --region='{region}' --model={model_id} --traffic-split=0=100 --machine-type='{machine_type}' --display-name='{model_name}'"

    run_command(command)


def undeploy_model(project, region, endpoint_id, model_id):
    command = f"gcloud ai endpoints undeploy-model {endpoint_id} --project='{project}' --region='{region}' --deployed-model-id='{model_id}'"

    run_command(command)


def delete_endpoint(project, region, endpoint_id):
    command = f"gcloud ai endpoints delete {endpoint_id} --project='{project}' --region='{region}'"

    run_command(command)


def check_repository(self, repository_name, region):
    command = f"gcloud artifacts repositories describe {repository_name} --location='{region}'"

    repository = run_command(command, capture_output=True)

    return repository


def deploy(
    project,
    region,
    model_name,
    model_env_vars=None,
    repository_name=None,
    repository_description=None,
    machine_type="n1_standard-2",
):
    # Start by defining the Google project
    set_project(project)

    # Next we check if we need to create a repository
    if repository_name:
        repository_description = check_repository(repository_name, region)

        # If we have a repository matching the name, we skip creation
        # Otherwise we create a new repository
        if not repository_description:
            create_artifact_repository(repository_name, region, repository_name)
        else:
            logger.info("Repository already exists, skipping creation")

    # Check if we have any existing models and endpoints using the names specified
    deployed_model_id = find_model(project, region, model_name)

    # If we have any existing models and endpoints, we delete them
    if deployed_model_id:
        deployed_endpoint_id = find_endpoint(project, region, model_name)

        undeploy_model(project, region, deployed_endpoint_id, deployed_model_id)

        delete_endpoint(project, region, deployed_endpoint_id)

    # THen we build our docker image
    create_build(project, repository_name, region, model_name)

    # Upload it to a model
    create_model(
        project, repository_name, region, model_name, model_name, model_env_vars
    )

    # Find the model's ID
    model_id = find_model(project, region, model_name)

    # Create an endpoint for the model
    create_model_endpoint(project, region, model_name)

    # Find the endpoint ID
    endpoint_id = find_endpoint(project, region, model_name)

    # And deploy the model to the endpoint.
    deploy_model(project, region, model_name, endpoint_id, model_id, machine_type)

    logger.info("Model deployed successfully")
