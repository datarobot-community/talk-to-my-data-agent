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

"""Identity resolution in `_initialize_session` (APP-6784).

Pre-fix, the `session_fastapi` cookie held `base64(uuid5(NAMESPACE_OID, email))` — unsigned
and computable from a victim's address alone — and the decoded value outranked the
`x-user-email` header. These tests pin the two properties that replaced it: the cookie is
never read, and the identity header resolves to its LAST value rather than its first.

Most drive `_initialize_session` directly: it touches no DataRobot client, no database and
no ASGI machinery, so a bare request stand-in exercises the trust boundary with nothing
stubbed. The exceptions are the properties that only exist a layer up — cookie issuance, the
removed account-uid override, and the account-fetch gate — which run the real
`session_middleware` through `TestClient` with DataRobot and database calls stubbed.
"""

from __future__ import annotations

import base64
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

import app.middleware as middleware

VICTIM = "victim@example.com"
ATTACKER = "attacker@example.com"
# Shaped like a DataRobot user id: 24 hex characters, not a uuid5.
OBJECT_ID = "5f4e3d2c1b0a998877665544"


def _uid(email: str) -> str:
    """The user_id derivation, verbatim from middleware."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, email))[:36]


def _request(*headers: tuple[str, str], **cookies: str) -> Any:
    """A request carrying exactly these headers, in this order.

    Names are lower-cased because that is what an ASGI server delivers. `Headers` lowers
    only the lookup key, so a raw entry stored as `X-User-Email` would match nothing and
    the test would fail for a reason that has nothing to do with the code under test.
    """
    request = MagicMock(spec=Request)
    request.headers = Headers(
        raw=[(k.lower().encode(), v.encode()) for k, v in headers],
    )
    request.cookies = cookies
    request.method = "GET"
    return request


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`session_store` is a module global; it would otherwise leak between tests.

    Cleared on the way out as well as in: without teardown the last test in this module
    leaves entries behind for every later module in the same pytest session.
    """
    monkeypatch.delenv("TEST_USER_EMAIL", raising=False)
    monkeypatch.delenv("APPLICATION_ID", raising=False)
    middleware.session_store.clear()
    yield
    middleware.session_store.clear()


@pytest.mark.asyncio
async def test_the_last_x_user_email_wins() -> None:
    """The proxy appends the authenticated address, so the last value is the real one.

    It uses `Header.Add` rather than `Del`+`Set`, so a client-supplied value survives
    alongside it and lands first — and first is what `.get()` returns. Reading the last
    value is what makes an injected header inert.
    """
    request = _request(("x-user-email", ATTACKER), ("x-user-email", VICTIM))

    # What the pre-fix read would have produced, asserted so the reason this test exists
    # cannot drift away from the code.
    assert request.headers.get("x-user-email") == ATTACKER

    result = await middleware._initialize_session(request)

    assert result.user_id == _uid(VICTIM)
    assert result.user_email == VICTIM
    assert list(middleware.session_store) == [_uid(VICTIM)]


@pytest.mark.asyncio
async def test_an_empty_authenticated_value_yields_no_identity() -> None:
    """A user whose platform email is empty must not be impersonable.

    The proxy adds the header unconditionally, so an empty `user.Email()` still appends an
    empty value after any injected one. `.get()` would return the attacker's address here
    and hand over that session outright.
    """
    request = _request(("x-user-email", VICTIM), ("x-user-email", ""))

    assert request.headers.get("x-user-email") == VICTIM

    result = await middleware._initialize_session(request)

    assert result.user_id is None
    assert result.user_email is None
    assert middleware.session_store == {}


@pytest.mark.asyncio
async def test_the_session_cookie_is_not_consulted() -> None:
    """The reported exploit: the cookie alone, with no header at all.

    `base64(uuid5(NAMESPACE_OID, victim))` is computable from an email address and nothing
    else. It used to be decoded straight into `user_id`.
    """
    forged = base64.b64encode(_uid(VICTIM).encode()).decode()
    request = _request(session_fastapi=forged)

    result = await middleware._initialize_session(request)

    assert result.user_id is None
    assert middleware.session_store == {}


@pytest.mark.asyncio
async def test_one_session_state_per_user() -> None:
    """The store is keyed by the derived id, so a user converges on one SessionState.

    Two instances would mean two AnalystDB handles over one user's DuckDB files, each with
    its own write lock, racing in the non-atomic save to persistent storage.
    """
    first = await middleware._initialize_session(_request(("x-user-email", VICTIM)))
    second = await middleware._initialize_session(_request(("x-user-email", VICTIM)))
    other = await middleware._initialize_session(_request(("x-user-email", ATTACKER)))

    assert first.session_state is second.session_state
    assert other.session_state is not first.session_state
    assert sorted(middleware.session_store) == sorted([_uid(VICTIM), _uid(ATTACKER)])


@pytest.mark.asyncio
@pytest.mark.parametrize("via_header", [True, False], ids=["header", "TEST_USER_EMAIL"])
async def test_user_id_derivation_is_pinned(
    via_header: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`user_id` is the partition key for every stored file, row and storage prefix.

    Pinned to a literal so no refactor can silently re-partition every existing user away
    from their data. Both entry points are covered: the deployed header and the local
    TEST_USER_EMAIL fallback, which nothing else exercises.
    """
    if via_header:
        request = _request(("x-user-email", VICTIM))
    else:
        monkeypatch.setenv("TEST_USER_EMAIL", VICTIM)
        request = _request()

    result = await middleware._initialize_session(request)

    assert result.user_id == "39f5e2e2-4a83-58aa-aae4-05f59d0ab2ba"
    assert result.user_email == VICTIM


@pytest.mark.asyncio
async def test_a_comma_joined_header_is_rejected_rather_than_partitioned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hop that folds repeated field-lines leaves attacker text in the last value.

    Deriving from it would key a partition on text the client chose, and `session_store`
    is never evicted — so varying the prefix would grow it, and the DuckDB files under it,
    without bound.
    """
    caplog.set_level(logging.ERROR, logger="app.middleware")
    request = _request(("x-user-email", f"{ATTACKER},{VICTIM}"))

    result = await middleware._initialize_session(request)

    assert result.user_id is None
    assert middleware.session_store == {}
    assert "comma-joined" in caplog.text


@pytest.mark.asyncio
async def test_a_client_supplied_comma_cannot_lock_a_user_out() -> None:
    """Rejecting comma-joined values must not hand a caller a denial-of-service.

    The proxy appends its value after anything the client sent, so a comma the client
    supplies is simply outvoted — the last value is still the authenticated address and
    carries no comma. The rejection therefore only fires when a hop coalesced the
    field-lines *after* the proxy, which is a topology fault and not client-reachable.
    """
    request = _request(
        ("x-user-email", f"{ATTACKER},spoofed@x"),
        ("x-user-email", VICTIM),
    )

    result = await middleware._initialize_session(request)

    assert result.user_id == _uid(VICTIM)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        pytest.param((("x-user-email", f"{ATTACKER},{VICTIM}"),), id="comma-joined"),
        pytest.param(
            (("x-user-email", VICTIM), ("x-user-email", "")), id="empty-authenticated"
        ),
    ],
)
async def test_a_rejected_header_does_not_fall_back_to_test_user_email(
    headers: tuple[tuple[str, str], ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected header must mean no identity, not the developer's identity.

    `TEST_USER_EMAIL` is set for `task dev` and `task test`, and the other tests in this
    module clear it — which hid this: both rejection paths used to fall through to the
    `elif test_user_email` branch and open the local developer's session. The fallback is
    for when the proxy sent no header at all, not for one that arrived and resolved to
    nothing.
    """
    monkeypatch.setenv("TEST_USER_EMAIL", "dev@example.com")

    result = await middleware._initialize_session(_request(*headers))

    assert result.user_id is None
    assert result.user_email is None
    assert middleware.session_store == {}


@pytest.mark.asyncio
async def test_test_user_email_still_applies_when_no_header_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: `task dev` has no proxy and sends no header."""
    monkeypatch.setenv("TEST_USER_EMAIL", "dev@example.com")

    result = await middleware._initialize_session(_request())

    assert result.user_id == _uid("dev@example.com")
    assert result.user_email == "dev@example.com"


@pytest.mark.asyncio
async def test_injected_duplicates_are_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The discard is silent to the caller, so the log line is the only operator signal."""
    caplog.set_level(logging.WARNING, logger="app.middleware")
    request = _request(("x-user-email", ATTACKER), ("x-user-email", VICTIM))

    await middleware._initialize_session(request)

    assert "Discarding 1 injected" in caplog.text


def _stub_account_fetch(
    monkeypatch: pytest.MonkeyPatch, account_info: dict[str, Any] | None = None
) -> MagicMock:
    """Keep the account/info call off the wire, returning `account_info` verbatim.

    conftest points DATAROBOT_ENDPOINT at a dummy host, so an unpatched fetch spends its
    full retry budget resolving DNS inside the event loop. The payload has to be a real dict
    rather than a bare MagicMock so that tests control what a restored account-uid override
    would read: `{}` is the shape that would let such a test pass for the wrong reason.
    """

    @contextmanager
    def no_token(request: Request, **kwargs: Any) -> Iterator[None]:
        yield

    dr_stub = MagicMock()
    dr_stub.client.get_client().get().json.return_value = account_info or {}
    # Configuring the chain above records calls on it; clear them so a test can assert on
    # whether the middleware fetched. `reset_mock()` leaves configured return values alone.
    dr_stub.reset_mock()
    monkeypatch.setattr(middleware, "use_user_token", no_token)
    monkeypatch.setattr(middleware, "dr", dr_stub)
    return dr_stub


def _middleware_app() -> FastAPI:
    """The real `session_middleware` over a route that reports what it was handed."""
    app = FastAPI()
    app.middleware("http")(middleware.session_middleware)

    @app.get("/whoami")
    async def whoami(request: Request) -> dict[str, Any]:
        return {"analyst_db": request.state.session._state.get("analyst_db")}

    return app


def test_no_session_fastapi_cookie_is_issued(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_initialize_session` cannot see this: issuance lives in `session_middleware`.

    Without a response-level assertion, restoring `response.set_cookie("session_fastapi",
    ...)` — a plausible merge-conflict resolution — reinstates the forgeable bearer token
    with the whole suite still green.
    """
    monkeypatch.setattr(middleware, "_initialize_database", AsyncMock())
    _stub_account_fetch(monkeypatch)

    with TestClient(_middleware_app()) as client:
        response = client.get("/whoami", headers={"x-user-email": VICTIM})

    assert response.status_code == 200
    cookies = response.headers.get_list("set-cookie")
    assert not any("session_fastapi" in c for c in cookies), cookies


def test_no_account_fetch_for_a_request_with_no_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch must follow resolved identity, not a raw read of the header.

    `headers.get("x-user-email")` returns the FIRST value, so on a request whose
    authenticated value was empty — identity rejected — it still reads the injected
    address as truthy and calls DataRobot for a user this request never resolved to.
    """
    monkeypatch.setattr(middleware, "_initialize_database", AsyncMock())
    dr_stub = _stub_account_fetch(monkeypatch, {"uid": "irrelevant"})

    with TestClient(_middleware_app()) as client:
        response = client.get(
            "/whoami",
            headers=[("x-user-email", ATTACKER), ("x-user-email", "")],
        )

    assert response.status_code == 200
    # A raw `.get()` would have seen ATTACKER here and fetched.
    assert dr_stub.client.get_client.return_value.get.call_count == 0


def test_a_header_matching_test_user_email_still_uses_the_visitor_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance is "did a header arrive", not "does the address differ".

    An earlier gate compared `user_email` to TEST_USER_EMAIL. A developer who sets that
    variable to the address the proxy sends then silently took the no-token branch and
    fetched with the ambient client, so scoped-token behaviour could not be reproduced
    locally.
    """
    monkeypatch.setenv("TEST_USER_EMAIL", VICTIM)
    monkeypatch.setattr(middleware, "_initialize_database", AsyncMock())
    _stub_account_fetch(monkeypatch)

    used_token_path = False

    @contextmanager
    def track(request: Request, **kwargs: Any) -> Iterator[None]:
        nonlocal used_token_path
        used_token_path = True
        yield

    monkeypatch.setattr(middleware, "use_user_token", track)

    with TestClient(_middleware_app()) as client:
        client.get("/whoami", headers={"x-user-email": VICTIM})

    assert used_token_path, "a real header must take the visitor-token branch"


def test_the_local_fallback_fetches_without_a_visitor_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`task dev` has no proxy and no visitor token to borrow.

    This is the only path that reaches the `elif user_email` arm, and the only
    configuration in which the `getlist` provenance conjunct changes anything. Without a
    test, deleting either leaves the suite green while local development silently loses
    its account info — or worse, calls `use_user_token`, which raises 401 when
    DR_CUSTOM_APP_EXTERNAL_URL is set and the bare `except` swallows it.
    """
    monkeypatch.setenv("TEST_USER_EMAIL", VICTIM)
    monkeypatch.setattr(middleware, "_initialize_database", AsyncMock())
    dr_stub = _stub_account_fetch(monkeypatch, {"uid": OBJECT_ID})

    used_token_path = False

    @contextmanager
    def track(request: Request, **kwargs: Any) -> Iterator[None]:
        nonlocal used_token_path
        used_token_path = True
        yield

    monkeypatch.setattr(middleware, "use_user_token", track)

    with TestClient(_middleware_app()) as client:
        client.get("/whoami")  # no x-user-email: the proxy is not in front of us

    assert not used_token_path, "there is no visitor token to borrow locally"
    assert dr_stub.client.get_client.return_value.get.call_count == 1


def test_account_info_uid_cannot_repartition_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The removed `datarobot_account_info["uid"]` fallback lived in `session_middleware`.

    Restoring it would hand `_initialize_database` a 24-hex DataRobot ObjectId instead of
    the uuid5, orphaning every user from the files stored under their real partition key.

    The ObjectId must reach the session the way it really would, from the account fetch and
    before `_initialize_database` runs. That precondition is asserted rather than assumed:
    without it the payload could be `{}`, a restored `.get("uid")` would read `None`, and
    this test would pass while the regression shipped.
    """
    opened: list[str] = []
    seen: list[Any] = []

    async def fake_init_db(
        request: Request, user_id: str, user_email: str | None = None
    ) -> None:
        opened.append(user_id)
        seen.append(request.state.session._state.get("datarobot_account_info"))

    monkeypatch.setattr(middleware, "_initialize_database", fake_init_db)
    _stub_account_fetch(monkeypatch, {"uid": OBJECT_ID})

    with TestClient(_middleware_app()) as client:
        client.get("/whoami", headers={"x-user-email": VICTIM})

    # Precondition: the ObjectId really was on the session before the DB was opened.
    assert seen == [{"uid": OBJECT_ID}], "account fetch did not deliver the uid"
    # The actual guarantee: it did not become the partition key.
    assert opened == [_uid(VICTIM)]


@pytest.mark.asyncio
async def test_initialize_database_sets_provided_user_email_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.session = middleware.SessionState({"analyst_db": None})

    mock_analyst_db = AsyncMock()
    mock_analyst_db.get_user_email = AsyncMock(return_value=None)
    mock_analyst_db.set_user_email = AsyncMock()

    async def mock_get_database(user_id: str) -> AsyncMock:
        assert user_id == "user-123"
        return mock_analyst_db

    monkeypatch.setattr(middleware, "get_database", mock_get_database)

    await middleware._initialize_database(
        request, user_id="user-123", user_email="new_user@example.com"
    )

    assert request.state.session.analyst_db is mock_analyst_db
    mock_analyst_db.get_user_email.assert_awaited_once()
    mock_analyst_db.set_user_email.assert_awaited_once_with("new_user@example.com")
