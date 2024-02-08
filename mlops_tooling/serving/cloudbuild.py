def write_cloudbuild_file(
    description: str,
    project: str,
    region: str,
    repository: str,
    image: str,
    model_name: str,
    model_id: str,
    endpoint_id: str,
    deployed_model_id: str,
    http_port: str,
    health_route: str,
    predict_route: str,
):

    base_file = f"""
    steps:
    - name: "gcr.io/cloud-builders/docker"
        id: "Build Docker image"
        args:
        [
            "build",
            "-t",
            "${{_BUILD_REGION}}-docker.pkg.dev/${{_BUILD_GOOGLE_PROJECT}}/${{_BUILD_REPOSITORY}}/${{_BUILD_IMAGE}}",
            "--build-arg",
            "_BUILD_AIP_HTTP_PORT=${{_BUILD_AIP_HTTP_PORT}}",
            "--build-arg",
            "_BUILD_AIP_HEALTH_ROUTE=${{_BUILD_AIP_HEALTH_ROUTE}}",
            "--build-arg",
            "_BUILD_AIP_PREDICT_ROUTE=${{_BUILD_AIP_PREDICT_ROUTE}}",
            ".",
        ]

    - name: "gcr.io/cloud-builders/docker"
        id: "Push Docker image"
        args:
        [
            "push",
            "${{_BUILD_REGION}}-docker.pkg.dev/${{_BUILD_GOOGLE_PROJECT}}/${{_BUILD_REPOSITORY}}/${{_BUILD_IMAGE}}",
        ]

    - name: "gcr.io/cloud-builders/gcloud"
        id: "Undeploy existing model from endpoint"
        args:
        [
            "ai",
            "endpoints",
            "undeploy-model",
            "${{_BUILD_ENDPOINT_ID}}",
            "--project=${{_BUILD_GOOGLE_PROJECT}}",
            "--region=${{_BUILD_REGION}}",
            "--deployed-model-id=${{_BUILD_DEPLOYED_MODEL_ID}}",
        ]
        allowFailure: true # In case the model is not already deployed

    - name: "gcr.io/cloud-builders/gcloud"
        id: "Delete existing endpoint"
        args:
        [
            "ai",
            "endpoints",
            "delete",
            "${{_BUILD_ENDPOINT_ID}}",
            "--project=${{_BUILD_GOOGLE_PROJECT}}",
            "--region=${{_BUILD_REGION}}",
        ]
        allowFailure: true # In case the endpoint does not exist

    - name: "gcr.io/cloud-builders/gcloud"
        id: "Upload model to Vertex AI"
        script: |
        #!/usr/bin/env bash

        # Define variables
        MODEL_EXISTS=$(gcloud ai models describe ${{_BUILD_MODEL_ID}} --region=${{_BUILD_REGION}} --format="value(name)" 2>/dev/null || echo "")

        # Check if the model exists
        if [ -z "$MODEL_EXISTS" ]; then
            # Model does not exist, upload without --parent-model
            gcloud ai models upload \
            --container-ports=${{_BUILD_AIP_HTTP_PORT}} \
            --container-predict-route=${{_BUILD_AIP_PREDICT_ROUTE}} \
            --container-health-route=${{_BUILD_AIP_HEALTH_ROUTE}} \
            --region=${{_BUILD_REGION}} \
            --display-name=${{_BUILD_MODEL_NAME}} \
            --container-image-uri=${{_BUILD_REGION}}-docker.pkg.dev/${{_BUILD_GOOGLE_PROJECT}}/${{_BUILD_REPOSITORY}}/${{_BUILD_IMAGE}} \
            --model-id=${{_BUILD_MODEL_ID}} \
            --version-aliases=default
        else
            # Model already exists, upload with --parent-model
            gcloud ai models upload \
            --container-ports=${{_BUILD_AIP_HTTP_PORT}} \
            --container-predict-route=${{_BUILD_AIP_PREDICT_ROUTE}} \
            --container-health-route=${{_BUILD_AIP_HEALTH_ROUTE}} \
            --region=${{_BUILD_REGION}} \
            --display-name=${{_BUILD_MODEL_NAME}} \
            --container-image-uri=${{_BUILD_REGION}}-docker.pkg.dev/${{_BUILD_GOOGLE_PROJECT}}/${{_BUILD_REPOSITORY}}/${{_BUILD_IMAGE}} \
            --model-id=${{_BUILD_MODEL_ID}} \
            --parent-model=projects/${{_BUILD_GOOGLE_PROJECT}}/locations/${{_BUILD_REGION}}/models/${{_BUILD_MODEL_ID}} \
            --version-aliases=default
        fi
        automapSubstitutions: true

    - name: "gcr.io/cloud-builders/gcloud"
        id: "Create an endpoint"
        args:
        [
            "ai",
            "endpoints",
            "create",
            "--project=${{_BUILD_GOOGLE_PROJECT}}",
            "--region=${{_BUILD_REGION}}",
            "--display-name=${{_BUILD_MODEL_NAME}}",
            "--endpoint-id=${{_BUILD_ENDPOINT_ID}}",
        ]

    - name: "gcr.io/cloud-builders/gcloud"
        id: "Deploy model to endpoint"
        args:
        [
            "ai",
            "endpoints",
            "deploy-model",
            "${{_BUILD_ENDPOINT_ID}}",
            "--project=${{_BUILD_GOOGLE_PROJECT}}",
            "--region=${{_BUILD_REGION}}",
            "--model=${{_BUILD_MODEL_ID}}",
            "--traffic-split=0=100",
            "--machine-type=n1-standard-2",
            "--display-name=${{_BUILD_IMAGE}}",
            "--deployed-model-id=${{_BUILD_DEPLOYED_MODEL_ID}}",
            "--enable-access-logging",
        ]

    options:
        logging: CLOUD_LOGGING_ONLY

    substitutions:
        _BUILD_DESCRIPTION: "{description}"
        _BUILD_GOOGLE_PROJECT: "{project}"
        _BUILD_REGION: "{region}"
        _BUILD_REPOSITORY: "{repository}"
        _BUILD_IMAGE: "{image}"
        _BUILD_MODEL_NAME: "{model_name}"
        _BUILD_MODEL_ID: "{model_id}"
        _BUILD_ENDPOINT_ID: "{endpoint_id}"
        _BUILD_DEPLOYED_MODEL_ID: "{deployed_model_id}"
        _BUILD_AIP_HTTP_PORT: "{http_port}"
        _BUILD_AIP_HEALTH_ROUTE: "{health_route}"
        _BUILD_AIP_PREDICT_ROUTE: "{predict_route}"
    """

    with open("cloudbuild.yaml", "w+") as f:
        f.writelines(base_file)
