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
import os
import uuid
from logging import getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Any, NamedTuple

import datarobot as dr
from core.analyst_db import AnalystDB
from core.datarobot_client import use_user_token
from core.telemetry import dr_user_id_var, otel
from fastapi import Request, Response

logger = getLogger(__name__)


class SessionInitializationResult(NamedTuple):
    """Structured return value for session initialization."""

    session_state: SessionState
    user_id: str | None
    user_email: str | None


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


# Module-level session store and lock. Keyed by `user_id`, derived in `_initialize_session`
# from the authenticated identity header. No cookie or other client-chosen value selects an
# entry.
#
# Entries are never evicted. That is deliberate: `routers/chats.py` hands `analyst_db` to
# background tasks for minute-long analyses, so dropping a live entry would let the next
# request open a second AnalystDB over the same DuckDB files. It holds one entry per
# distinct authenticated user per pod, which is why `_initialize_session` refuses to derive
# an id from any header value it cannot attribute to the proxy — an unbounded key space
# here would be an unbounded memory and disk leak.
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
        user_metadata_db_name="user_metadata.db",
        use_persistent_storage=bool(os.environ.get("APPLICATION_ID")),
    )
    return analyst_db


@otel.trace
@otel.meter
async def _initialize_session(
    request: Request,
) -> SessionInitializationResult:
    """Resolve the caller's identity and hand back their session state."""
    test_user_email = os.environ.get("TEST_USER_EMAIL", "")

    if test_user_email and os.environ.get("APPLICATION_ID"):
        logger.fatal("Test email set on a deployed instance.")
        raise RuntimeError("Test eail set on a deployed instance.")

    session_state = SessionState(
        {
            "datarobot_account_info": None,
            "datarobot_api_scoped_token": None,
            "analyst_db": None,
        }
    )

    # LAST value, not `.get()`. The upstream proxy adds this header with `Header.Add`
    # rather than `Del`+`Set`, so a client-supplied value survives alongside the
    # authenticated one and lands FIRST — which is what `.get()` returns. The last value is
    # the one written by the hop closest to this app, i.e. the proxy.
    #
    # That holds only while every request reaches us through that proxy, and the two ways it
    # can fail need different fixes. An *injected* value arriving through the proxy is closed
    # for good by an unconditional `Del` upstream, tracked separately. A request that
    # *bypasses* the proxy never reaches that `Del`: it carries a single client-controlled
    # value and no authentication at all, and only network isolation — or authentication in
    # this app — would close that one.
    #
    # Do not change the derivation below. `user_id` is the partition key for every per-user
    # DuckDB file and every persistent-storage key prefix, so any change to it orphans
    # existing users from their datasets, chats and dictionaries.
    emails = request.headers.getlist("x-user-email")
    if len(emails) > 1:
        logger.warning("Discarding %d injected x-user-email value(s).", len(emails) - 1)

    authenticated = emails[-1] if emails else ""
    if "," in authenticated:
        # A hop folded repeated field-lines into one comma-joined value, so the last value
        # is no longer just the proxy's. Rejected rather than used: deriving from the
        # joined string would mint a partition keyed on attacker-chosen text, and since
        # `session_store` is never evicted, varying the prefix would grow it without bound.
        logger.error("Rejecting comma-joined x-user-email; cannot identify the caller.")
        return SessionInitializationResult(session_state, None, None)

    user_id: str | None = None
    user_email: str | None = None
    if authenticated:
        user_email = authenticated
        user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user_email))[:36]
    elif not emails and test_user_email:
        # Local development only, and only when the proxy sent nothing at all. A header
        # that arrived but resolved to nothing means the proxy did identify the caller —
        # as someone with no address — so falling back to the developer's identity there
        # would turn a rejected request into an authenticated one.
        user_email = test_user_email
        user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, test_user_email))[:36]

    # No identity: no session to reuse and no database to open. `deps.py` turns this into
    # a 400. Nothing is stored, so an unauthenticated caller cannot grow `session_store`.
    if user_id is None:
        return SessionInitializationResult(session_state, None, None)

    async with session_lock:
        existing_session = session_store.get(user_id)
        if existing_session is not None:
            session_state = existing_session
        else:
            session_store[user_id] = session_state

    return SessionInitializationResult(session_state, user_id, user_email)


async def _initialize_database(
    request: Request, user_id: str, user_email: str | None = None
) -> None:
    """Initialize per-user database in the session if not already initialized."""
    if (
        not hasattr(request.state.session, "analyst_db")
        or request.state.session.analyst_db is None
    ):
        async with session_lock:
            analyst_db = await get_database(user_id)
            request.state.session.analyst_db = analyst_db
            if user_email and user_email != await analyst_db.get_user_email():
                await analyst_db.set_user_email(user_email)


async def session_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Middleware to manage user sessions."""
    request_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    if request.method in request_methods:
        dr_user_id_var.set(request.headers.get("x-user-id"))

        # Initialize the session
        session_init = await _initialize_session(request)
        user_id: str | None = session_init.user_id
        user_email = session_init.user_email
        request.state.session = session_init.session_state

        if not request.state.session.datarobot_account_info:
            request.state.session.datarobot_account_info = {}
            # Gate on the identity `_initialize_session` resolved, not on a fresh
            # `headers.get("x-user-email")`: that reads the FIRST value, so a request whose
            # authenticated value was empty — no identity — would still fetch, for whoever
            # the injected first value named.
            #
            # `getlist` is only asked whether a header arrived, never for its value — that
            # is enough to tell a proxied request from the local TEST_USER_EMAIL fallback,
            # and unlike comparing `user_email` to TEST_USER_EMAIL it stays right when a
            # developer sets that variable to the same address the header carries.
            try:
                if user_email and request.headers.getlist("x-user-email"):
                    # do not try to fetch user info for prob requests
                    with use_user_token(request):
                        reply = dr.client.get_client().get("account/info/")
                        account_info = reply.json()
                    request.state.session.datarobot_account_info = account_info
                elif user_email:
                    # Local development: there is no visitor token to borrow.
                    reply = dr.client.get_client().get("account/info/")
                    account_info = reply.json()
                    request.state.session.datarobot_account_info = account_info
            except Exception as e:
                logger.info(f"Error fetching account info: {e}")

        # Initialize database in the session
        if user_id:
            await _initialize_database(request, user_id, user_email=user_email)

    # No session cookie is issued. A `session_fastapi` cookie still held by a browser is
    # ignored rather than cleared. Clearing it would mean emitting a Set-Cookie on every
    # response to delete a name at path "/" that this app no longer owns and that sibling
    # template-derived apps still use — cost with no benefit, since an unrecognised cookie
    # is already inert here.
    response: Response = await call_next(request)

    return response
