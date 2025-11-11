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

# LLM Model Configuration
ALTERNATIVE_LLM_BIG = "datarobot-deployed-llm"
ALTERNATIVE_LLM_SMALL = "datarobot-deployed-llm"
DEFAULT_LLM_GATEWAY_MODEL = "azure/gpt-4o"
DEFAULT_LLM_GATEWAY_MODEL_SMALL = "azure/gpt-4o-mini"

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

TIKTOKEN_ENCODING_MAP = {
    ALTERNATIVE_LLM_BIG: DEFAULT_TIKTOKEN_ENCODING,
    ALTERNATIVE_LLM_SMALL: DEFAULT_TIKTOKEN_ENCODING,
    DEFAULT_LLM_GATEWAY_MODEL: DEFAULT_TIKTOKEN_ENCODING,
    DEFAULT_LLM_GATEWAY_MODEL_SMALL: DEFAULT_TIKTOKEN_ENCODING,
}

# Error Messages
VALUE_ERROR_MESSAGE = "Input data cannot be empty (no dataset provided)"
