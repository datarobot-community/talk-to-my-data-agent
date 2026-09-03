# Copyright 2025 DataRobot, Inc.
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
"""
This configuration option is right choice when you already have an LLM from Azure, Bedrock,
Anthropic, Vertex, etc. This way you can monitor and scale your LLM directly with the added
benefits of the DataRobot platform such as governance, guard models, controlled API access,
and monitoring.
"""

import os

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.common.feature_flags import check_feature_flag_set
from datarobot_pulumi_utils.common.llm_validation import verify_llm
from datarobot_pulumi_utils.pulumi import export
from datarobot_pulumi_utils.pulumi.llm_credentials import get_runtime_values
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironments
from datarobot_pulumi_utils.schema.llms import ensure_datarobot_prefix

from . import use_case

__all__ = [
    "app_runtime_parameters",
    "custom_model_runtime_parameters",
    "default_model",
    "llm_application_name",
    "llm_resource_name",
]

REQUIRED_FEATURE_FLAGS = {
    "ENABLE_MLOPS": True,
    "ENABLE_CUSTOM_INFERENCE_MODEL": True,
    "ENABLE_PUBLIC_NETWORK_ACCESS_FOR_ALL_CUSTOM_MODELS": True,
    "ENABLE_MLOPS_TEXT_GENERATION_TARGET_TYPE": True,
}

llm_application_name: str = "llm"
llm_resource_name: str = "[llm]"
default_model: str = ensure_datarobot_prefix(
    os.environ.get("LLM_DEFAULT_MODEL", "datarobot/azure/gpt-5-mini")
)
default_llm_id: str = os.environ.get(
    "LLM_DEFAULT_LLM_ID",
    "azure-openai-gpt-5-mini",  # External LLM ID from the Playground
)
default_llm_friendly_name: str = os.environ.get(
    "LLM_DEFAULT_LLM_NAME",
    "Claude Sonnet 4.5",  # Shown in the Web UI
)
default_use_builder_api_token = os.environ.get("USE_BUILDER_API_TOKEN", "false")

check_feature_flag_set(REQUIRED_FEATURE_FLAGS)
# get_runtime_values() must run before verify_llm() below: it hydrates this provider's
# credential env vars (e.g. cross-populating AZURE_API_KEY from OPENAI_API_KEY) as a side
# effect, and verify_llm() relies on that hydration having already happened.
llm_credential_runtime_params = get_runtime_values(
    default_llm_id, resource_suffix=llm_resource_name
)
# Smoke-test the external provider directly (before the DataRobot deployment exists) using the
# credentials in the environment. verify_llm() strips the datarobot/ prefix so LiteLLM addresses
# the provider directly. See https://docs.litellm.ai/docs/providers for the model string each
# provider expects (e.g. bedrock/..., vertex_ai/...). For Azure, LiteLLM addresses the model by
# its Azure *deployment name*, so prefer OPENAI_API_DEPLOYMENT_ID when it is set.
verify_model = default_model.removeprefix("datarobot/")
azure_deployment_id = os.getenv("OPENAI_API_DEPLOYMENT_ID")
if verify_model.startswith("azure/") and azure_deployment_id:
    verify_model = f"azure/{azure_deployment_id}"
verify_llm(verify_model)

playground = datarobot.Playground(
    use_case_id=use_case.id,
    resource_name="LLM Playground " + llm_resource_name,
)

llm_blueprint = datarobot.LlmBlueprint(
    resource_name="LLM Blueprint " + llm_resource_name,
    playground_id=playground.id,
    llm_id=default_llm_id,
    llm_settings=datarobot.LlmBlueprintLlmSettingsArgs(
        max_completion_length=2048,
        temperature=0.1,
        top_p=None,
    ),
)

llm_custom_model = datarobot.CustomModel(
    resource_name="LLM Custom Model " + llm_resource_name,
    name="LLM Custom Model " + llm_resource_name,
    target_name="resultText",
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=1,
    base_environment_id=RuntimeEnvironments.PYTHON_312_MODERATIONS.value.id,
    use_case_ids=[use_case.id],
    source_llm_blueprint_id=llm_blueprint.id,
    runtime_parameter_values=llm_credential_runtime_params,
)

if prediction_environment_id := os.environ.get(
    "DATAROBOT_DEFAULT_PREDICTION_ENVIRONMENT"
):
    pulumi.info(f"Using existing prediction environment '{prediction_environment_id}'")

    prediction_environment = datarobot.PredictionEnvironment.get(
        id=prediction_environment_id,
        resource_name="LLM Prediction Environment "
        + llm_resource_name
        + " [PRE-EXISTING]",
    )
else:
    prediction_environment = datarobot.PredictionEnvironment(
        resource_name="LLM Prediction Environment " + llm_resource_name,
        platform=dr.enums.PredictionEnvironmentPlatform.DATAROBOT_SERVERLESS,
    )

# Register the custom model
llm_registered_model = datarobot.RegisteredModel(
    resource_name="LLM Registered Model " + llm_resource_name,
    custom_model_version_id=llm_custom_model.version_id,
    name="LLM Registered Model " + llm_resource_name,
    use_case_ids=[use_case.id],
)

# Deploy the registered model
llm_deployment = datarobot.Deployment(
    resource_name="LLM Blueprint Deployment " + llm_resource_name,
    registered_model_version_id=llm_registered_model.version_id,
    prediction_environment_id=prediction_environment.id,
    label=f"LLM Deployment [{PROJECT_NAME}] " + llm_resource_name,
    use_case_ids=[use_case.id],
    association_id_settings=datarobot.DeploymentAssociationIdSettingsArgs(
        column_names=["association_id"],
        auto_generate_id=False,
        required_in_prediction_requests=True,
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True,
    ),
    predictions_settings=datarobot.DeploymentPredictionsSettingsArgs(
        min_computes=0, max_computes=2
    ),
    opts=pulumi.ResourceOptions(replace_on_changes=["registered_model_version_id"]),
)

app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="LLM_DEPLOYMENT_ID",
        type="string",
        value=llm_deployment.id,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="USE_DATAROBOT_LLM_GATEWAY",
        type="string",
        value="0",
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="LLM_DEFAULT_MODEL",
        type="string",
        value=default_model,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="LLM_DEFAULT_MODEL_FRIENDLY_NAME",
        type="string",
        value=default_llm_friendly_name,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="USE_BUILDER_API_TOKEN",
        type="string",
        value=default_use_builder_api_token,
    ),
]
custom_model_runtime_parameters = [
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="LLM_DEPLOYMENT_ID",
        type="string",
        value=llm_deployment.id,
    ),
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="USE_DATAROBOT_LLM_GATEWAY",
        type="string",
        value="0",
    ),
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="LLM_DEFAULT_MODEL",
        type="string",
        value=default_model,
    ),
]

# TODO(APP-5859): Move datarobot_url to af-component-base infra/__init__.py
datarobot_url = (
    os.getenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")
    .rstrip("/")
    .removesuffix("/api/v2")
)
rag_playground_url = pulumi.Output.format(
    "{0}/usecases/{1}/playgrounds/{2}/comparison/chats",
    datarobot_url,
    use_case.id,
    playground.id,
)
deployment_url = pulumi.Output.format(
    "{0}/console-nextgen/deployments/{1}/overview",
    datarobot_url,
    llm_deployment.id,
)

pulumi.export("Deployment ID " + llm_resource_name, llm_deployment.id)
pulumi.export("Deployment Console " + llm_resource_name, deployment_url)
export("LLM_DEPLOYMENT_ID", llm_deployment.id)
export("USE_DATAROBOT_LLM_GATEWAY", "0")
export("LLM_DEFAULT_MODEL", default_model)
export("LLM_DEFAULT_MODEL_FRIENDLY_NAME", default_llm_friendly_name)
export("USE_BUILDER_API_TOKEN", default_use_builder_api_token)
pulumi.export("RAG Playground URL " + llm_resource_name, rag_playground_url)
