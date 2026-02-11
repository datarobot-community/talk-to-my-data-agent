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

"""Middleware components for the Data Analyst API."""

from __future__ import annotations

import asyncio
import base64
import os
import uuid
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import datarobot as dr
from fastapi import Request, Response

from core.analyst_db import AnalystDB
from core.datarobot_client import use_user_token

from .telemetry import otel

logger = getLogger(__name__)


class SessionState(object):
    """Session state container for user-specific data."""

    _state: dict[str, Any]

    def __init__(self, state: dict[str, Any] | None = None):
        if state is None:
            state = {}
        super().__setattr__("_state", state)

    def __setattr__(self, key: Any, value: Any) -> None:
        self._state[key] = value

    def __getattr__(self, key: Any) -> Any:
        try:
            return self._state[key]
        except KeyError:
            message = "'{}' object has no attribute '{}'"
            raise AttributeError(message.format(self.__class__.__name__, key))

    def __delattr__(self, key: Any) -> None:
        del self._state[key]

    def update(self, state: dict[str, Any]) -> None:
        self._state.update(state)


# Module-level session store and lock
session_store: dict[str, SessionState] = {}
session_lock = asyncio.Lock()


async def get_database(user_id: str) -> AnalystDB:
    """Create an AnalystDB instance for the given user."""
    tmp = gettempdir()
    analyst_db = await AnalystDB.create(
        user_id=user_id,
        db_path=Path(tmp),
        dataset_db_name="datasets.db",
        chat_db_name="chat.db",
        data_source_db_name="datasources.db",
        user_recipe_db_name="recipe.db",
        use_persistent_storage=bool(os.environ.get("APPLICATION_ID")),
    )
    return analyst_db


@otel.trace
@otel.meter
async def _initialize_session(
    request: Request,
) -> tuple[SessionState, str | None, str | None]:
    """Initialize the session state and return the session ID and user ID."""
    test_user_email = os.environ.get("TEST_USER_EMAIL", "")

    if test_user_email and os.environ.get("APPLICATION_ID"):
        logger.fatal("Test email set on a deployed instance.")
        raise RuntimeError("Test eail set on a deployed instance.")

    # Create a new session state with default values
    session_state = SessionState()
    empty_session_state: dict[str, Any] = {
        "datarobot_account_info": None,
        "datarobot_api_scoped_token": None,
        "analyst_db": None,
    }
    session_state.update(deepcopy(empty_session_state))

    # Try to get user ID from cookie
    user_id = None

    session_fastapi_cookie = request.cookies.get("session_fastapi")
    if session_fastapi_cookie:
        try:
            user_id = base64.b64decode(session_fastapi_cookie.encode()).decode()
        except Exception:
            pass  # If decoding fails, continue without user_id

    # Generate a new user ID if needed
    new_user_id = None
    email_header = request.headers.get("x-user-email")
    if email_header:
        new_user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, email_header))[:36]
    elif test_user_email:
        new_user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, test_user_email))[:36]

    # Determine session ID
    session_id = None
    if session_fastapi_cookie:
        session_id = session_fastapi_cookie
    elif new_user_id:
        session_id = base64.b64encode(new_user_id.encode()).decode()

    # Get or create session in store
    if session_id:
        async with session_lock:
            existing_session = session_store.get(session_id)
            if existing_session:
                return existing_session, session_id, user_id or new_user_id
            else:
                session_store[session_id] = session_state

    return session_state, session_id, user_id or new_user_id


async def _initialize_database(request: Request, user_id: str) -> None:
    """Initialize per-user database in the session if not already initialized."""
    if (
        not hasattr(request.state.session, "analyst_db")
        or request.state.session.analyst_db is None
    ):
        async with session_lock:
            request.state.session.analyst_db = await get_database(user_id)


def _set_session_cookie(
    response: Response,
    user_id: str | None,
    session_id: str,
    session_fastapi_cookie: str | None,
) -> None:
    """Set the session cookie if needed."""
    if user_id and not session_fastapi_cookie:
        encoded_uid = base64.b64encode(user_id.encode()).decode()
        response.set_cookie(key="session_fastapi", value=encoded_uid, httponly=True)
    elif not session_fastapi_cookie and not user_id:
        response.set_cookie(key="session_fastapi", value=session_id, httponly=True)


async def session_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Middleware to manage user sessions."""
    request_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    session_id: str | None = None
    user_id: str | None = None

    if request.method in request_methods:
        # Initialize the session
        session_state, session_id, user_id = await _initialize_session(request)
        request.state.session = session_state

        if not request.state.session.datarobot_account_info:
            request.state.session.datarobot_account_info = {}
            try:
                if request.headers.get("x-user-email"):
                    # do not try to fetch user info for prob requests
                    with use_user_token(request):
                        reply = dr.client.get_client().get("account/info/")
                        account_info = reply.json()
                    request.state.session.datarobot_account_info = account_info
                elif os.environ.get("TEST_USER_EMAIL", ""):
                    reply = dr.client.get_client().get("account/info/")
                    account_info = reply.json()
                    request.state.session.datarobot_account_info = account_info
            except Exception as e:
                logger.info(f"Error fetching account info: {e}")

        dr_uid = request.state.session.datarobot_account_info.get("uid")
        if session_id is None and dr_uid is not None:
            session_id = base64.b64encode(dr_uid.encode()).decode()
            user_id = dr_uid

        # Initialize database in the session
        if user_id:
            await _initialize_database(request, user_id)

    # Process the request
    response: Response = await call_next(request)

    if request.method in request_methods:
        # Set session cookie if needed
        if session_id:
            _set_session_cookie(
                response, user_id, session_id, request.cookies.get("session_fastapi")
            )

    return response
