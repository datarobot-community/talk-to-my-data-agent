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

import os

import datarobot as dr
import pulumi
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.custom_models import (
    CustomModelArgs,
    DeploymentArgs,
    RegisteredModelArgs,
)
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironments

from utils.schema import LLMDeploymentSettings

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")

custom_model_args = CustomModelArgs(
    resource_name=f"Data Analyst Proxy LLM Custom Model [{PROJECT_NAME}]",
    name=f"Data Analyst Proxy LLM Custom Model [{PROJECT_NAME}]",
    target_name=LLMDeploymentSettings().target_feature_name,
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=2,
    base_environment_id=RuntimeEnvironments.PYTHON_312_MODERATIONS.value.id,
    opts=pulumi.ResourceOptions(delete_before_replace=True),
)

registered_model_args = RegisteredModelArgs(
    resource_name=f"Data Analyst Proxy LLM Registered Model [{PROJECT_NAME}]",
)

deployment_args = DeploymentArgs(
    resource_name=f"Data Analyst Proxy LLM Deployment [{PROJECT_NAME}]",
    label=f"Data Analyst Proxy LLM Deployment [{PROJECT_NAME}]",
)
