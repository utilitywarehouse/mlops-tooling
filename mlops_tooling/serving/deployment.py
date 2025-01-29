import argparse
import re
import subprocess

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


def create_model(
    project,
    repository,
    region,
    image,
    model_name,
    model_id,
    parent_model=False,
    env_vars=None,
):
    command = f"gcloud ai models upload --container-ports=80 --container-predict-route='/predict' --container-health-route='/health' --region='{region}' --display-name='{model_name}' --container-image-uri='{region}-docker.pkg.dev/{project}/{repository}/{image}' --version-aliases=default"

    if parent_model:
        command += (
            f" --parent-model=projects/{project}/locations/{region}/models/{model_id}"
        )

    if env_vars:
        add_ons = ",".join(env_vars)
        command += add_ons

    run_command(command)


def create_model_endpoint(project, region, model_name, endpoint_id=None):
    command = f"gcloud ai endpoints create --project={project} --region={region} --display-name={model_name}"

    if endpoint_id:
        command += f"--endpoint-id={endpoint_id}"

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
    project,
    region,
    endpoint_id,
    model_id,
    model_name,
    deployed_model_id,
    machine_type="n1_standard-2",
):
    command = f"gcloud ai endpoints deploy-model {endpoint_id} --project='{project}' --region='{region}' --model={model_id} --traffic-split=0=100 --machine-type='{machine_type}' --display-name='{model_name}' --enable-access-logging"

    if deployed_model_id:
        command += f"--deployed-model-id={deployed_model_id}"

    run_command(command)


def undeploy_model(project, region, endpoint_id, model_id):
    command = f"gcloud ai endpoints undeploy-model {endpoint_id} --project='{project}' --region='{region}' --deployed-model-id='{model_id}'"

    run_command(command)


def delete_endpoint(project, region, endpoint_id):
    command = f"gcloud ai endpoints delete {endpoint_id} --project='{project}' --region='{region}'"

    run_command(command)


def check_repository(repository_name, region):
    command = f"gcloud artifacts repositories describe {repository_name} --location='{region}'"

    repository = run_command(command, capture_output=True)

    return repository


def deploy(
    project,
    region,
    model_name,
    model_id=None,
    endpoint_id=None,
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

        endpoint_exists = deployed_endpoint_id != endpoint_id

        if deployed_endpoint_id != endpoint_id:
            delete_endpoint(project, region, deployed_endpoint_id)

    # THen we build our docker image
    create_build(project, repository_name, region, model_name)

    has_parent_model = deployed_model_id == model_id if model_id else False

    create_model(
        project,
        repository_name,
        region,
        model_name,
        model_name,
        model_id,
        has_parent_model,
        model_env_vars,
    )

    if not model_id:
        # Find the model's ID
        model_id = find_model(project, region, model_name)

    if not endpoint_exists:
        # Create an endpoint for the model
        create_model_endpoint(project, region, model_name)

        # Find the endpoint ID
        endpoint_id = find_endpoint(project, region, model_name)

    # And deploy the model to the endpoint.
    deploy_model(project, region, model_name, endpoint_id, model_id, machine_type)

    logger.info("Model deployed successfully")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Deploy models using GCP commands.")
    parser.add_argument("--project", required=True, help="Google Cloud project name")
    parser.add_argument("--region", required=True, help="GCP region")
    parser.add_argument("--model-name", required=True, help="Name of the model")
    parser.add_argument(
        "--model-id", required=False, help="Specific model ID requested"
    )
    parser.add_argument(
        "--endpoint-id", required=False, help="Specific endpoint ID requested"
    )
    parser.add_argument(
        "--model-env-vars",
        nargs="*",
        required=False,
        help="Environment variables for the model",
    )
    parser.add_argument(
        "--repository-name", required=False, help="Name of the artifact repository"
    )
    parser.add_argument(
        "--repository-description",
        required=False,
        help="Description of the artifact repository",
    )
    parser.add_argument(
        "--machine-type",
        default="n1_standard-2",
        required=False,
        help="Machine type for deployment",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    deploy(
        args.project,
        args.region,
        args.model_name,
        args.model_id,
        args.endpoint_id,
        args.model_env_vars,
        args.repository_name,
        args.repository_description,
        args.machine_type,
    )
