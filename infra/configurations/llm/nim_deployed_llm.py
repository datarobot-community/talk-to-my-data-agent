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
Choose this option when you already have a DataRobot NIM deployment and its deployment ID.
It attaches that existing NIM deployment to the playground and use case. For a non-NIM custom
model deployment choose "deployed_llm.py"; for a registered model that needs to be wrapped in an
LLM Blueprint and deployed choose "registered_model.py".
"""

import os

import pulumi
import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.common.feature_flags import check_feature_flag_set
from datarobot_pulumi_utils.common.llm_validation import verify_llm
from datarobot_pulumi_utils.pulumi import export
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
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

# Read the deployment ID from the app-prefixed var, falling back to the bare NIM_DEPLOYMENT_ID.
# Use `or` (not a dict-get default) so the required bare lookup is only evaluated when the
# prefixed var is absent.
LLM_NIM_DEPLOYMENT_ID = (
    os.environ.get("LLM_NIM_DEPLOYMENT_ID") or os.environ["NIM_DEPLOYMENT_ID"]
)

llm_application_name: str = "llm"
llm_resource_name: str = "[llm]"
# NIM serves a specific model, so this is normally provided. It defaults to an inert placeholder
# only as a last resort; set LLM_DEFAULT_MODEL to the real NIM model name.
default_model: str = ensure_datarobot_prefix(
    os.environ.get("LLM_DEFAULT_MODEL", DEPLOYED_LLM_PLACEHOLDER_MODEL)
)

# Verify everything is working
check_feature_flag_set(REQUIRED_FEATURE_FLAGS)
verify_llm(model_id=f"{default_model}", deployment_id=LLM_NIM_DEPLOYMENT_ID)

playground = datarobot.Playground(
    use_case_id=use_case.id,
    resource_name=f"LLM Playground [{PROJECT_NAME}] " + llm_resource_name,
)
proxy_llm_deployment = datarobot.Deployment.get(
    resource_name="Existing LLM Deployment", id=LLM_NIM_DEPLOYMENT_ID
)
prediction_environment = datarobot.PredictionEnvironment.get(
    resource_name="Existing LLM Prediction Environment",
    id=proxy_llm_deployment.prediction_environment_id,
)
app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="NIM_DEPLOYMENT_ID",
        type="string",
        value=proxy_llm_deployment.id,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="LLM_DEFAULT_MODEL",
        type="string",
        value=default_model,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key="USE_DATAROBOT_LLM_GATEWAY",
        type="string",
        value="0",
    ),
]
custom_model_runtime_parameters = [
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="NIM_DEPLOYMENT_ID",
        type="string",
        value=proxy_llm_deployment.id,
    ),
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="LLM_DEFAULT_MODEL",
        type="string",
        value=default_model,
    ),
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="USE_DATAROBOT_LLM_GATEWAY",
        type="string",
        value="0",
    ),
]

pulumi.export("Deployment ID " + llm_resource_name, proxy_llm_deployment.id)
export("NIM_DEPLOYMENT_ID", proxy_llm_deployment.id)
export("LLM_DEFAULT_MODEL", default_model)
export("USE_DATAROBOT_LLM_GATEWAY", "0")
