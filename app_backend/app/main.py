# Copyright 2024 DataRobot, Inc.
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

import json
import logging
import os
from typing import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from utils.rest_api import app

# Configure logging to filter out the health check logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out "GET /" health check logs
        return "GET / HTTP/1.1" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

SCRIPT_NAME = os.environ.get("SCRIPT_NAME", "")
SERVE_STATIC_FRONTEND = os.getenv("SERVE_STATIC_FRONTEND", "True").casefold() == "true"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
base_router = APIRouter()


@base_router.get("/_dr_env.js")
async def get_env() -> Response:
    NOTEBOOK_ID = os.getenv("NOTEBOOK_ID", "")
    app_base_url = os.getenv("BASE_PATH", "")
    if not app_base_url and NOTEBOOK_ID:
        app_base_url = f"notebook-sessions/{NOTEBOOK_ID}"

    env_vars = {
        "APP_BASE_URL": app_base_url,
        "API_PORT": os.getenv("PORT"),
        "DATAROBOT_ENDPOINT": os.getenv("DATAROBOT_ENDPOINT", ""),
        "IS_STATIC_FRONTEND": SERVE_STATIC_FRONTEND,
        "USE_DATAROBOT_LLM_GATEWAY": os.getenv("USE_DATAROBOT_LLM_GATEWAY", "false"),
    }
    js = f"window.ENV = {json.dumps(env_vars)};"
    return Response(content=js, media_type="application/javascript")


# Serve runtime env script from nested paths as well to support deep reloads
@base_router.get("/{tail:path}/_dr_env.js")
async def get_env_catch_all(tail: str) -> Response:
    return await get_env()


# Serve SPA index.html for known app routes (deep reload support)
if SERVE_STATIC_FRONTEND:

    @base_router.get(f"{SCRIPT_NAME}/")
    @base_router.get(f"{SCRIPT_NAME}/data")
    @base_router.get(f"{SCRIPT_NAME}/data/{{dataId:path}}")
    @base_router.get(f"{SCRIPT_NAME}/chats")
    @base_router.get(f"{SCRIPT_NAME}/chats/{{chat_id:path}}")
    async def serve_spa() -> Response:
        return FileResponse(
            os.path.join(STATIC_DIR, "index.html"), media_type="text/html"
        )


app.include_router(base_router)

if SERVE_STATIC_FRONTEND:
    # Normalize '/assets/' from deep routes (e.g., /data/...) back to '/assets/...'
    # because built HTML uses relative asset URLs.
    def create_static_path_middleware() -> Callable[
        [Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]
    ]:
        async def static_path_normalize(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            path = request.scope.get("path")
            if isinstance(path, str):
                # If already canonical, leave it. If nested, collapse to canonical tail.
                if path.startswith("/assets/"):
                    return await call_next(request)
                if "/assets/" in path:
                    assets_idx = path.find("/assets/")
                    request.scope["path"] = path[assets_idx:]
            return await call_next(request)

        return static_path_normalize

    app.middleware("http")(create_static_path_middleware())

    # Important to be last so that we fall back to the static files if the route is not found
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
