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
import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.pulumi import export

__all__ = [
    "custom_model_runtime_parameters",
    "app_runtime_parameters",
    "default_model",
    "llm_application_name",
    "llm_resource_name",
]

__all__ = [
    "llm_application_name",
    "llm_resource_name",
]

llm_application_name: str = "llm"
llm_resource_name: str = "[llm]"
default_model: str = os.environ.get("LLM_DEFAULT_MODEL", "azure/openai-gpt-5-mini")
default_llm_id: str = os.environ.get(
    "LLM_DEFAULT_LLM_ID",
    "azure/openai-gpt-5-mini",  # External LLM ID from the Playground
)
default_llm_friendly_name: str = os.environ.get(
    "LLM_DEFAULT_LLM_NAME",
    "Azure OpenAI GPT-5 Mini",  # Shown in the Web UI
)


app_runtime_parameters = [
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
]
custom_model_runtime_parameters = [
    datarobot.CustomModelRuntimeParameterValueArgs(
        key="LLM_DEFAULT_MODEL",
        type="string",
        value=default_model,
    ),
]

export("LLM_DEFAULT_MODEL", default_model)
export("LLM_DEFAULT_MODEL_FRIENDLY_NAME", default_llm_friendly_name)
export("USE_DATAROBOT_LLM_GATEWAY", "0")
