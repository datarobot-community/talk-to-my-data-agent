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

"""Common dependencies for the Data Analyst API."""

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from core.analyst_db import AnalystDB
from core.logging_helper import get_logger

logger = get_logger()


def get_initialized_db(request: Request) -> AnalystDB:
    """Dependency to provide the initialized database."""
    if (
        not hasattr(request.state.session, "analyst_db")
        or request.state.session.analyst_db is None
    ):
        if not request.headers.get("x-user-email"):
            logger.error("x-user-email is required in order to initialize the database")
        raise HTTPException(
            status_code=400,
            detail="Database not initialized.",
        )

    return cast(AnalystDB, request.state.session.analyst_db)
