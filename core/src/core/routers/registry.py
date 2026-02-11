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

"""Router for Data Registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from core.api import list_registry_datasets
from core.datarobot_client import use_user_token
from core.schema import DataRegistryDataset

router = APIRouter(prefix="/registry", tags=["registry"])


# Make this sync as the DR requests are synchronous
@router.get("/datasets")
def get_registry_datasets(
    request: Request, remote: bool = False, limit: int = 100
) -> list[DataRegistryDataset]:
    """Return all registry datasets

    Args:
        request (Request): HTTP request
        remote (bool, optional): Whether to fetch remote datasets
        limit (int, optional): Maximum number of datasets to return

    Returns:
        list[DataRegistryDataset]: List of registry datasets
    """
    with use_user_token(request):
        return list_registry_datasets(remote=remote, limit=limit)
