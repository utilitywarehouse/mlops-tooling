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


def deploy(
    project,
    region,
    model_name,
    model_env_vars=None,
    repository_name=None,
    repository_description=None,
    machine_type="n1_standard-2",
):
    set_project(project)

    if repository_name:
        create_artifact_repository(
            project, repository_name, region, repository_description
        )

    create_build(project, repository_name, region, model_name)

    create_model(
        project, repository_name, region, model_name, model_name, model_env_vars
    )

    model_id = find_model(project, region, model_name)

    create_model_endpoint(project, region, model_name)

    endpoint_id = find_endpoint(project, region, model_name)

    deploy_model(project, region, model_name, endpoint_id, model_id, machine_type)

    logger.info("Model deployed successfully")
