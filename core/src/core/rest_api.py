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

"""Main FastAPI application factory for the Data Analyst API."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.types import Lifespan

from core.logging_helper import get_logger
from core.middleware import session_middleware
from core.routers import (
    chats_router,
    database_router,
    datasets_router,
    dictionaries_router,
    external_data_stores_router,
    registry_router,
    supported_types_router,
    user_router,
)

from .telemetry import otel

logger = get_logger()


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """Generate custom OpenAPI schema with security definitions."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_app(
    lifespan: Lifespan[FastAPI] | None = None,
    title: str = "Data Analyst API",
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured FastAPI application instance
    """
    app = FastAPI(
        title="Data Analyst API",
        description="""
        An intelligent API for data analysis that provides capabilities including:
        - Dataset management (upload CSV/Excel files, connect to databases, access the Data Registry)
        - Data cleansing and standardization
        - Data dictionary creation and management
        - Chat-based data analysis conversations
        - Python code generation
        - Chart creation
        - Business insights generation

        Available endpoint groups:
        - /api/v1/registry: Access Data Registry datasets
        - /api/v1/database: Database connection and table management
        - /api/v1/datasets: Upload, retrieve, and manage datasets
        - /api/v1/dictionaries: Manage data dictionaries
        - /api/v1/chats: Create and manage chat conversations for data analysis

        The API uses OpenAI's GPT models for intelligent analysis and response generation.
        """,
        version="1.0.0",
        contact={"name": "API Support", "email": "support@example.com"},
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        lifespan=lifespan,
        debug=True,  # Stack traces will be exposed for 500 responses
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Add session middleware
    app.middleware("http")(session_middleware)

    # Set custom OpenAPI schema
    app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]

    # Get script name from environment
    script_name = os.environ.get("SCRIPT_NAME", "")
    prefix = f"{script_name}/api/v1"

    # Register all routers
    app.include_router(registry_router, prefix=prefix)
    app.include_router(database_router, prefix=prefix)
    app.include_router(datasets_router, prefix=prefix)
    app.include_router(dictionaries_router, prefix=prefix)
    app.include_router(chats_router, prefix=prefix)
    app.include_router(user_router, prefix=prefix)
    app.include_router(external_data_stores_router, prefix=prefix)
    app.include_router(supported_types_router, prefix=prefix)

    # Initialize telemetry on application startup
    otel.log_application_start()

    # Setup auto-instrumentation for FastAPI
    # This will automatically trace all incoming HTTP requests
    otel.instrument_fastapi_app(app)

    return app
