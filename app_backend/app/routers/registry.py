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

from core.api import list_registry_datasets
from core.datarobot_client import use_user_token
from core.schema import RegistryDatasets
from fastapi import APIRouter, Request

router = APIRouter(prefix="/registry", tags=["registry"])


# Make this sync as the DR requests are synchronous
@router.get("/datasets")
def get_registry_datasets(request: Request, limit: int = 100) -> RegistryDatasets:
    """Return the local and remote registry datasets.

    Both listings come from a single catalog walk. They used to be two separate
    requests distinguished by a `remote` flag, which the frontend issued
    concurrently — two simultaneous AI Catalog searches per user, which the
    platform rejects with a 409 (AECO-44).

    Args:
        request (Request): HTTP request
        limit (int, optional): Maximum number of datasets to return per listing

    Returns:
        RegistryDatasets: The local and remote registry dataset listings
    """
    with use_user_token(request, allow_use_builder_token=True):
        return list_registry_datasets(limit=limit)
