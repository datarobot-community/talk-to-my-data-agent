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

"""Generic reverse proxy to the DataRobot API.

The ``@datarobot/connectivity`` Browse Data component talks to the DataRobot
platform (data registry, data connections, structured connectors). The browser
cannot call the platform directly — it must go through this app so requests are
authenticated with the signed-in user's scoped token. This router forwards any
request to ``{DATAROBOT_ENDPOINT}/{path}`` and streams the response back.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx
from core.datarobot_client import (
    FILE_API_CONNECT_TIMEOUT,
    FILE_API_READ_TIMEOUT,
    get_visitors_token,
)
from core.logging_helper import get_logger
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = get_logger()

router = APIRouter(prefix="/proxy", tags=["DataRobot Proxy"])

# Inbound client headers that are safe to forward upstream.
_FORWARDED_REQUEST_HEADERS = frozenset(
    [
        "content-type",
        "accept",
        "x-datarobot-identity-token",
    ]
)

# Hop-by-hop and encoding headers that must not be forwarded to the client.
# content-length is dropped too: httpx may decode a compressed upstream body, so
# the upstream length need not match the bytes we stream — let Starlette frame it.
_SKIP_RESPONSE_HEADERS = frozenset(
    [
        "transfer-encoding",
        "connection",
        "keep-alive",
        "content-encoding",
        "content-length",
    ]
)


def _build_proxy_headers(request: Request, token: str) -> dict[str, str]:
    """Build outbound headers for the upstream DataRobot request.

    Authorizes with the visiting user's scoped token so upstream acts as the
    signed-in user (and can reach that user's externalDataStores /
    externalConnectors catalog endpoints) — never the app builder's service
    token.
    """
    headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
    for name, value in request.headers.items():
        if name.lower() in _FORWARDED_REQUEST_HEADERS:
            headers[name] = value
    return headers


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Strip hop-by-hop headers before forwarding the upstream response."""
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _SKIP_RESPONSE_HEADERS
    }


@router.api_route(
    "/datarobot/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_to_datarobot(
    request: Request,
    path: str,
) -> StreamingResponse:
    """Reverse proxy to ``{DATAROBOT_ENDPOINT}/{path}``.

    Authenticates as the visiting user via their scoped DataRobot token. The
    app builder's service token is intentionally never used — Browse Data must
    act as the signed-in user so it surfaces that user's own resources.

    Example::

        GET /api/v1/proxy/datarobot/api/v2/externalDataStores/
        →   GET {DATAROBOT_ENDPOINT}/api/v2/externalDataStores/
    """
    endpoint = os.environ.get("DATAROBOT_ENDPOINT")
    if not endpoint:
        raise HTTPException(status_code=500, detail="DATAROBOT_ENDPOINT is not set.")

    # allow_use_builder_token only returns the service token when the app is
    # configured with USE_BUILDER_API_TOKEN=true (set by the dev task for local
    # runs). In deployed apps that flag is off, so this stays visitor-only.
    token = get_visitors_token(request, allow_use_builder_token=True)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="API token required. Please authenticate with DataRobot.",
            headers={"x-datarobot-auth-required": "true"},
        )

    base = endpoint.rstrip("/")
    # If the caller already included /api/v2 in the path, strip it from the
    # endpoint to avoid a doubled prefix (endpoint may end with /api/v2).
    if path.startswith("api/v2"):
        base = base.removesuffix("/api/v2")
    target_url = f"{base}/{path}"
    if request.query_params:
        target_url = f"{target_url}?{request.query_params}"

    proxy_headers = _build_proxy_headers(request, token)
    body = await request.body()

    logger.info("Proxy %s %s", request.method, target_url)

    # Do NOT follow redirects — a reverse proxy should return 3xx responses to
    # the caller as-is. Following them silently can hide auth failures (e.g.
    # DataRobot redirecting an unauthenticated request to a login page).
    #
    # Use the file API's generous timeouts (default 180s, env-overridable)
    # instead of httpx's 5s default — Browse Data / registry calls and large
    # uploads can be slow, and a 5s cap turns them into spurious 502s.
    client = httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(FILE_API_READ_TIMEOUT, connect=FILE_API_CONNECT_TIMEOUT),
    )
    upstream_request = client.build_request(
        method=request.method,
        url=target_url,
        headers=proxy_headers,
        content=body,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        logger.warning(
            "Proxy upstream error: %s %s — %s", request.method, target_url, exc
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "Proxy %s %s → %d (%s)",
        request.method,
        target_url,
        upstream.status_code,
        upstream.headers.get("content-type", "no content-type"),
    )

    async def _body_generator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.HTTPError as exc:
            logger.error("Proxy stream error for %s: %s", target_url, exc)
            raise
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = _filter_response_headers(upstream.headers)
    # Tell the custom-apps websocket-proxy that a 401 from us is a real upstream
    # auth failure, not a signal that this app is broken. Without this header the
    # websocket-proxy rewrites 401 → 302 → 503, hiding the real status. APP-5913.
    if upstream.status_code == 401:
        response_headers["x-datarobot-auth-required"] = "true"

    return StreamingResponse(
        _body_generator(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
