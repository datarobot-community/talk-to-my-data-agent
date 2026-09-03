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
Choose this option when you have an existing registered model (for example a TextGen model)
that is not yet deployed. It deploys the registered model, wraps that deployment in an LLM
Blueprint via a CustomModelLlmValidation, then registers and deploys the blueprint for full
DataRobot governance and monitoring.
"""

import os

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.common.feature_flags import check_feature_flag_set
from datarobot_pulumi_utils.common.llm_validation import verify_llm
from datarobot_pulumi_utils.pulumi import export
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironments
from datarobot_pulumi_utils.schema.llms import (
    DEPLOYED_LLM_PLACEHOLDER_MODEL,
    ensure_datarobot_prefix,
)

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

TEXTGEN_REGISTERED_MODEL_ID = os.environ["TEXTGEN_REGISTERED_MODEL_ID"]

llm_application_name: str = "llm"
llm_resource_name: str = "[llm]"
# The blueprint deployment routes by its deployment ID; the model string is only a label
# (the endpoint ignores it), so it defaults to an inert placeholder. Set
# LLM_DEFAULT_MODEL to the real model name if you want datarobot-genai to
# match provider-specific reasoning parameters.
default_model: str = ensure_datarobot_prefix(
    os.environ.get("LLM_DEFAULT_MODEL", DEPLOYED_LLM_PLACEHOLDER_MODEL)
)
default_use_builder_api_token = os.environ.get("USE_BUILDER_API_TOKEN", "false")

# Verify the feature flags are available
check_feature_flag_set(REQUIRED_FEATURE_FLAGS)

playground = datarobot.Playground(
    use_case_id=use_case.id,
    resource_name=f"LLM Playground [{PROJECT_NAME}]" + llm_resource_name,
)

# Pull in the registered model
proxy_llm_registered_model = datarobot.RegisteredModel.get(
    resource_name="Existing TextGen Registered Model",
    id=TEXTGEN_REGISTERED_MODEL_ID,
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

# Create the deployment for the passed in registered model
proxy_llm_deployment = datarobot.Deployment(
    resource_name=f"LLM Deployment [{PROJECT_NAME}]",
    registered_model_version_id=proxy_llm_registered_model.version_id,
    prediction_environment_id=prediction_environment.id,
    label=f"Data Analyst LLM Deployment [{PROJECT_NAME}]",
    use_case_ids=[use_case.id],
    opts=pulumi.ResourceOptions(replace_on_changes=["registered_model_version_id"]),
)

# Use Pulumi apply to verify the registered model LLM once deployed
proxy_llm_deployment.id.apply(  # type: ignore[missing-argument]
    lambda id: verify_llm(model_id=f"{default_model}", deployment_id=id)  # type: ignore[invalid-argument-type]
)

# Make a LLM Blueprint from the deployed registered model
proxy_llm_validation = datarobot.CustomModelLlmValidation(
    resource_name="LLM Blueprint Validation " + llm_resource_name,
    chat_model_id=default_model.removeprefix("datarobot/"),
    deployment_id=proxy_llm_deployment.id,
    use_case_id=use_case.id,
)
llm_blueprint = datarobot.LlmBlueprint(
    resource_name="LLM Blueprint " + llm_resource_name,
    custom_model_llm_settings=datarobot.LlmBlueprintCustomModelLlmSettingsArgs(
        validation_id=proxy_llm_validation.id,
    ),
    llm_id="custom-model",
    playground_id=playground.id,
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
    runtime_parameter_values=[],
)

# Register the custom model from the LLM Blueprint
llm_blueprint_registered_model = datarobot.RegisteredModel(
    resource_name="LLM Registered Model " + llm_resource_name,
    custom_model_version_id=llm_custom_model.version_id,
    name="LLM Registered Model " + llm_resource_name,
    use_case_ids=[use_case.id],
)

# Deploy the LLM Blueprint Registered Model
llm_deployment = datarobot.Deployment(
    resource_name="LLM Blueprint Deployment " + llm_resource_name,
    registered_model_version_id=llm_blueprint_registered_model.version_id,
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
    # The app talks to the governed blueprint deployment (llm_deployment).
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
        value=proxy_llm_registered_model.name,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="USE_BUILDER_API_TOKEN",
        type="string",
        value=default_use_builder_api_token,
    ),
]
custom_model_runtime_parameters = [
    # The blueprint custom model wraps and calls the underlying proxy deployment. It must not
    # reference llm_deployment.id (its own downstream deployment) or Pulumi would form a cycle.
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="LLM_DEPLOYMENT_ID",
        type="string",
        value=proxy_llm_deployment.id,
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

# Export the deployment the app actually uses (the governed blueprint deployment), matching
# app_runtime_parameters above.
pulumi.export("Deployment ID " + llm_resource_name, llm_deployment.id)
export("LLM_DEPLOYMENT_ID", llm_deployment.id)
export("USE_DATAROBOT_LLM_GATEWAY", "0")
export("LLM_DEFAULT_MODEL", default_model)
export("LLM_DEFAULT_MODEL_FRIENDLY_NAME", proxy_llm_registered_model.name)
export("USE_BUILDER_API_TOKEN", default_use_builder_api_token)
