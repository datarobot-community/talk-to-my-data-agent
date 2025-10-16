# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from typing import Any, Optional

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.common.feature_flags import check_feature_flags
from datarobot_pulumi_utils.common.urls import get_deployment_url
from datarobot_pulumi_utils.pulumi.custom_model_deployment import CustomModelDeployment
from datarobot_pulumi_utils.pulumi.proxy_llm_blueprint import ProxyLLMBlueprint
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.apps import CustomAppResourceBundles
from datarobot_pulumi_utils.schema.llms import LLMs

sys.path.append("..")

from settings_main import PROJECT_ROOT

from infra import (
    settings_app_infra,
    settings_generative,
)
from infra.app_frontend import app_frontend
from infra.components.dr_credential import (
    get_credential_runtime_parameter_values,
    get_database_credentials,
    get_llm_credentials,
)
from infra.settings_database import DATABASE_CONNECTION_TYPE
from infra.settings_proxy_llm import CHAT_MODEL_NAME
from utils.resources import (
    app_env_name,
    llm_deployment_env_name,
)
from utils.schema import AppInfra


def fetch_and_prepare_app_resources(source_id: str) -> Optional[dict[str, Any]]:
    """
    Fetch resource configuration from a CustomApplicationSource entity
    and prepare it for CustomApplication creation.

    Args:
        source_id: The ID of the CustomApplicationSource to fetch resources from

    Returns:
        Dictionary containing resource configuration compatible with CustomApplication,
        or None if not configured
    """
    try:
        source = dr.CustomApplicationSource.get(source_id)
        pulumi.info(f"Fetched CustomApplicationSource: {source.name} (ID: {source.id})")

        resources = source.get_resources()
        if resources:
            pulumi.info(f"Found resources in source: {resources}")
            # Prepare resources in the format expected by CustomApplication
            app_resources = {
                "resource_label": resources.get("resource_label"),
                "replicas": resources.get("replicas"),
            }
            # Optional fields - only include if present
            if resources.get("session_affinity") is not None:
                app_resources["session_affinity"] = resources.get("session_affinity")
            if resources.get("service_web_requests_on_root_path") is not None:
                app_resources["service_web_requests_on_root_path"] = resources.get(
                    "service_web_requests_on_root_path"
                )
            return app_resources
        else:
            pulumi.warn("No resources configured in CustomApplicationSource")
            return None
    except Exception as e:
        pulumi.warn(f"Failed to fetch resources from CustomApplicationSource: {e}")
        return None


def create_resources_args(
    source_id: str,
) -> Optional[datarobot.CustomApplicationResourcesArgs]:
    """
    Fetch resources from source and convert to Pulumi CustomApplicationResourcesArgs.

    Args:
        source_id: The ID of the CustomApplicationSource

    Returns:
        CustomApplicationResourcesArgs if resources exist, None otherwise
    """
    resources = fetch_and_prepare_app_resources(source_id)
    if resources:
        return datarobot.CustomApplicationResourcesArgs(**resources)
    return None


TEXTGEN_DEPLOYMENT_ID = os.environ.get("TEXTGEN_DEPLOYMENT_ID")
TEXTGEN_REGISTERED_MODEL_ID = os.environ.get("TEXTGEN_REGISTERED_MODEL_ID")
USE_DATAROBOT_LLM_GATEWAY = (
    os.environ.get("USE_DATAROBOT_LLM_GATEWAY", "false").lower() == "true"
)


# Check if OpenAI credentials are available (same logic as in utils/api.py)
def has_openai_credentials() -> bool:
    """Check if OpenAI credentials are available in environment variables."""
    # Check for the essential OpenAI environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")

    if api_key and api_base:
        pulumi.info("OpenAI credentials detected in environment variables")
        return True
    return False


# 1. If `LLMs.DEPLOYED_LLM` is set, an already deployed DataRobot-hosted LLM deployment i.e.,
#    NVIDIA NIM, Cohere, Shared LLM Deployment, or other custom model of the text gen type.
# 2. If `LLMs.DEPLOYED_LLM` is unset and OpenAI Credentials are set, spin up an LLM Blueprint deployed
#    with OpenAI credentials
# 3. If `LLMs.DEPLOYED_LLM` is unset and OpenAI Credentials are not set, it check if pay as you go
#    pricing is enabled and spin up an LLM Blueprint with DataRobot credentials
HAS_OPENAI_CREDS = has_openai_credentials()
USE_LLM_GATEWAY = USE_DATAROBOT_LLM_GATEWAY and not HAS_OPENAI_CREDS

if USE_LLM_GATEWAY:
    pulumi.info("Using LLM Gateway - OpenAI credentials will not be required")
    check_feature_flags(
        PROJECT_ROOT / "infra" / "feature_flag_requirements_llm_gateway.yaml"
    )

if settings_generative.LLM == LLMs.DEPLOYED_LLM:
    pulumi.info(f"{TEXTGEN_DEPLOYMENT_ID=}")
    pulumi.info(f"{TEXTGEN_REGISTERED_MODEL_ID=}")
    if TEXTGEN_DEPLOYMENT_ID is not None:
        pulumi.info(f"Using existing deployment '{TEXTGEN_DEPLOYMENT_ID}'")
        if TEXTGEN_REGISTERED_MODEL_ID is not None:
            pulumi.warn("TEXTGEN_REGISTERED_MODEL_ID will be ignored")
            TEXTGEN_REGISTERED_MODEL_ID = None
    if TEXTGEN_REGISTERED_MODEL_ID is not None:
        pulumi.info(f"Using existing registered model '{TEXTGEN_REGISTERED_MODEL_ID}'")


check_feature_flags(PROJECT_ROOT / "infra" / "feature_flag_requirements.yaml")

with open(
    settings_app_infra.application_path / "app_infra.json", "w"
) as infra_selection:
    infra_selection.write(
        AppInfra(
            database=DATABASE_CONNECTION_TYPE,
            llm=settings_generative.LLM.name,  # Always write the actual LLM name
        ).model_dump_json()
    )

if "DATAROBOT_DEFAULT_USE_CASE" in os.environ:
    use_case_id = os.environ["DATAROBOT_DEFAULT_USE_CASE"]
    pulumi.info(f"Using existing use case '{use_case_id}'")
    use_case = datarobot.UseCase.get(
        id=use_case_id,
        resource_name="Data Analyst Use Case [PRE-EXISTING]",
    )
else:
    use_case = datarobot.UseCase(
        resource_name=f"Data Analyst Use Case [{PROJECT_NAME}]",
        description="Use case for Data Analyst application",
    )

prediction_environment = datarobot.PredictionEnvironment(
    resource_name=f"Data Analyst Prediction Environment [{PROJECT_NAME}]",
    platform=dr.enums.PredictionEnvironmentPlatform.DATAROBOT_SERVERLESS,
)

if not USE_LLM_GATEWAY:
    llm_credential = get_llm_credentials(settings_generative.LLM)

    llm_runtime_parameter_values = get_credential_runtime_parameter_values(
        llm_credential, "llm"
    )

playground = datarobot.Playground(
    use_case_id=use_case.id,
    **settings_generative.playground_args.model_dump(),
)

if settings_generative.LLM == LLMs.DEPLOYED_LLM:
    if TEXTGEN_REGISTERED_MODEL_ID is not None:
        proxy_llm_registered_model = datarobot.RegisteredModel.get(
            resource_name="Existing TextGen Registered Model",
            id=TEXTGEN_REGISTERED_MODEL_ID,
        )

        proxy_llm_deployment = datarobot.Deployment(
            resource_name=f"Data Analyst LLM Deployment [{PROJECT_NAME}]",
            registered_model_version_id=proxy_llm_registered_model.version_id,
            prediction_environment_id=prediction_environment.id,
            label=f"Data Analyst LLM Deployment [{PROJECT_NAME}]",
            use_case_ids=[use_case.id],
            opts=pulumi.ResourceOptions(
                replace_on_changes=["registered_model_version_id"]
            ),
        )
    elif TEXTGEN_DEPLOYMENT_ID is not None:
        proxy_llm_deployment = datarobot.Deployment.get(
            resource_name="Existing LLM Deployment", id=TEXTGEN_DEPLOYMENT_ID
        )
    else:
        raise ValueError(
            "Either TEXTGEN_REGISTERED_MODEL_ID or TEXTGEN_DEPLOYMENT_ID have to be set in `.env`"
        )
    llm_blueprint = ProxyLLMBlueprint(
        use_case_id=use_case.id,
        playground_id=playground.id,
        proxy_llm_deployment_id=proxy_llm_deployment.id,
        chat_model_name=CHAT_MODEL_NAME,
        **settings_generative.llm_blueprint_args.model_dump(mode="python"),
    )

elif settings_generative.LLM != LLMs.DEPLOYED_LLM:
    llm_blueprint = datarobot.LlmBlueprint(  # type: ignore[assignment]
        playground_id=playground.id,
        **settings_generative.llm_blueprint_args.model_dump(),
    )

llm_custom_model = datarobot.CustomModel(
    **settings_generative.custom_model_args.model_dump(exclude_none=True),
    use_case_ids=[use_case.id],
    source_llm_blueprint_id=llm_blueprint.id,
    runtime_parameter_values=(
        []
        if settings_generative.LLM == LLMs.DEPLOYED_LLM or USE_LLM_GATEWAY
        else llm_runtime_parameter_values
    ),
)

llm_deployment = CustomModelDeployment(
    resource_name=f"Chat Agent Deployment [{PROJECT_NAME}]",
    use_case_ids=[use_case.id],
    custom_model_version_id=llm_custom_model.version_id,
    registered_model_args=settings_generative.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_generative.deployment_args,
)

app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=llm_deployment_env_name,
        type="deployment",
        value=llm_deployment.id,
    ),
]

db_credential = get_database_credentials(DATABASE_CONNECTION_TYPE)

db_runtime_parameter_values = get_credential_runtime_parameter_values(
    db_credential, "db"
)
app_runtime_parameters += db_runtime_parameter_values  # type: ignore[arg-type]

# Only pass LLM Gateway runtime parameters if we're actually using LLM Gateway
# This prevents confusion when OpenAI credentials override LLM Gateway settings
if USE_LLM_GATEWAY:
    app_runtime_parameters.append(
        datarobot.ApplicationSourceRuntimeParameterValueArgs(
            key="USE_DATAROBOT_LLM_GATEWAY",
            type="string",
            value="true",
        )
    )

app_source = datarobot.ApplicationSource(
    files=app_frontend.stdout.apply(
        lambda _: settings_app_infra.get_app_files(
            runtime_parameter_values=app_runtime_parameters
        )
    ),
    runtime_parameter_values=app_runtime_parameters,
    resources=datarobot.ApplicationSourceResourcesArgs(
        resource_label=CustomAppResourceBundles.CPU_XL.value.id,
    ),
    **settings_app_infra.app_source_args,
)

# Create the custom application with resources dynamically fetched from the source
# Use Pulumi's .apply() to handle the async nature of the source ID
app = datarobot.CustomApplication(
    resource_name=settings_app_infra.app_resource_name,
    source_version_id=app_source.version_id,
    use_case_ids=[use_case.id],
    allow_auto_stopping=True,
    resources=app_source.id.apply(create_resources_args),  # type: ignore[arg-type]
)

pulumi.export(llm_deployment_env_name, llm_deployment.id)
pulumi.export(
    settings_generative.deployment_args.resource_name,
    llm_deployment.id.apply(get_deployment_url),
)
# Export LLM Gateway configuration for visibility
pulumi.export(
    "USE_DATAROBOT_LLM_GATEWAY", "true" if USE_DATAROBOT_LLM_GATEWAY else "false"
)

# App output
pulumi.export(app_env_name, app.id)
pulumi.export(
    settings_app_infra.app_resource_name,
    app.application_url,
)
