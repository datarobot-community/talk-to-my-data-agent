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
from __future__ import annotations

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.custom_models import (
    CustomModelArgs,
    DeploymentArgs,
    RegisteredModelArgs,
)
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironments
from datarobot_pulumi_utils.schema.llms import (
    LLMBlueprintArgs,
    LLMs,
    LLMSettings,
    PlaygroundArgs,
)

from utils.schema import LLMDeploymentSettings

LLM = LLMs.AZURE_OPENAI_GPT_4_O

custom_model_args = CustomModelArgs(
    resource_name=f"Generative Analyst Custom Model [{PROJECT_NAME}]",
    name="Generative Analyst Assistant",  # built-in QA app uses this as the AI's name
    target_name=LLMDeploymentSettings().target_feature_name,
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=2,
    base_environment_id=RuntimeEnvironments.PYTHON_312_MODERATIONS.value.id,
    opts=pulumi.ResourceOptions(delete_before_replace=True),
)

registered_model_args = RegisteredModelArgs(
    resource_name=f"Generative Analyst Registered Model [{PROJECT_NAME}]",
)


deployment_args = DeploymentArgs(
    resource_name=f"Generative Analyst Deployment [{PROJECT_NAME}]",
    label=f"Generative Analyst Deployment [{PROJECT_NAME}]",
    association_id_settings=datarobot.DeploymentAssociationIdSettingsArgs(
        column_names=["association_id"],
        auto_generate_id=False,
        required_in_prediction_requests=True,
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True,
    ),
    predictions_settings=(
        datarobot.DeploymentPredictionsSettingsArgs(min_computes=0, max_computes=2)
    ),
)

playground_args = PlaygroundArgs(
    resource_name=f"Generative Analyst Playground [{PROJECT_NAME}]",
)

llm_blueprint_args = LLMBlueprintArgs(
    resource_name=f"Generative Analyst LLM Blueprint [{PROJECT_NAME}]",
    llm_id=LLM.name,
    llm_settings=LLMSettings(
        max_completion_length=2048,
        temperature=0.1,
    ),
)
