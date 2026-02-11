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
from typing import Final

from core.config import Config


def _normalize_model_name(raw_model: str) -> str:
    """
    Add datarobot as a provider and handle any other provider string fixes for
    litellm
    """
    # if the model is already a datarobot prefix, return it as is
    if raw_model.startswith("datarobot/"):
        return raw_model
    # fallback to datarobot provider
    return f"datarobot/{raw_model}"


# LLM Model Configuration
def get_llm_model(preferred_model: str = Config().llm_default_model) -> str:
    """Get the configured LLM model with datarobot/ prefix."""
    return _normalize_model_name(preferred_model)


# If you are using the LLM Gateway, and you want to use different
# mdoels for different use cases, you can set those here. Otherwise, they will default
# to the same model specified in the config.
ALTERNATIVE_LLM_BIG = get_llm_model()
ALTERNATIVE_LLM_SMALL = get_llm_model()

# Dictionary Generation Configuration
DICTIONARY_BATCH_SIZE = 10
DICTIONARY_PARALLEL_BATCH_SIZE = 2
DICTIONARY_TIMEOUT = 45.0

# Dataset Size Limits
MAX_REGISTRY_DATASET_SIZE = 400e6  # aligns to 400MB set in streamlit config.toml
REGISTRY_DATASET_SIZE_CUTOFF: Final[float] = (
    200e6  # at 200MB we move from downloading to analyzing remotely with dataset
)
DISK_CACHE_LIMIT_BYTES = 512e6

# Token and Context Limits
MAX_CSV_TOKENS = 50000  # limit for data analyst csv sended to llm
MODEL_CONTEXT_WINDOW = 128000  # GPT-4o context window
CONTEXT_WARNING_THRESHOLD = int(MODEL_CONTEXT_WINDOW * 0.8)

# Tiktoken Encoding Configuration
DEFAULT_TIKTOKEN_ENCODING = "o200k_base"


def get_tiktoken_encoding_map() -> dict[str, str]:
    """Get tiktoken encoding map with current model configuration."""
    model = get_llm_model()
    return {
        model: DEFAULT_TIKTOKEN_ENCODING,
    }


TIKTOKEN_ENCODING_MAP = get_tiktoken_encoding_map()

# Error Messages
VALUE_ERROR_MESSAGE = "Input data cannot be empty (no dataset provided)"
