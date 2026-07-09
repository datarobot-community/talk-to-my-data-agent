# Copyright 2026 DataRobot, Inc.
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

from pyinstrument import Profiler
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PyInstrumentMiddleware:
    """ASGI middleware that profiles a request when ?profile=1 is present.

    Only active when the app is started with PROFILING_ENABLED=true.
    Add it to the stack before registering routes so the full request
    path is covered, including other middleware.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        if request.query_params.get("profile") != "1":
            await self.app(scope, receive, send)
            return

        async def null_send(message: Message) -> None:
            pass

        profiler = Profiler(async_mode="enabled")
        profiler.start()
        try:
            await self.app(scope, receive, null_send)
        finally:
            profiler.stop()

        response = HTMLResponse(profiler.output_html())
        await response(scope, receive, send)
