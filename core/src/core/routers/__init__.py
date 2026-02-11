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

"""Routers for the Data Analyst API."""

from core.routers.chats import router as chats_router
from core.routers.database import router as database_router
from core.routers.datasets import router as datasets_router
from core.routers.dictionaries import router as dictionaries_router
from core.routers.external_data_stores import (
    external_data_stores_router,
    supported_types_router,
)
from core.routers.registry import router as registry_router
from core.routers.user import router as user_router

__all__ = [
    "chats_router",
    "database_router",
    "datasets_router",
    "dictionaries_router",
    "external_data_stores_router",
    "registry_router",
    "supported_types_router",
    "user_router",
]
