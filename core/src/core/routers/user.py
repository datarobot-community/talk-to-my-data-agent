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

"""Router for user account endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request

from core.datarobot_client import get_visitors_token

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/datarobot-account")
async def get_datarobot_account(
    request: Request,
) -> dict[str, Any]:
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    scoped_token = get_visitors_token(request)

    truncated_api_token = None
    if api_token:
        truncated_api_token = f"****{api_token[-4:]}"

    truncated_scoped_token = None
    if scoped_token:
        truncated_scoped_token = f"****{scoped_token[-4:]}"

    return {
        "datarobot_account_info": request.state.session.datarobot_account_info,
        "datarobot_api_token": truncated_api_token,
        "datarobot_api_scoped_token": truncated_scoped_token,
    }


@router.post("/datarobot-account")
async def store_datarobot_account(
    request: Request,
) -> dict[str, Any]:
    """Store DataRobot account information."""
    request_data = await request.json()

    if "api_token" in request_data and request_data["api_token"]:
        request.state.session.datarobot_api_scoped_token = request_data["api_token"]

    return {"success": True}
