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

"""
Data Analyst specific telemetry implementation.

This module extends the base telemetry with application-specific
logging, metrics, and tracing functions for the Data Analyst app.
Includes auto-instrumentation for FastAPI and requests library.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

from .base_telemetry import BaseTelemetry

if TYPE_CHECKING:
    from fastapi import FastAPI

# Optional imports for auto-instrumentation
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:
    FastAPIInstrumentor = None  # type: ignore[assignment, misc]

try:
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
except ImportError:
    RequestsInstrumentor = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class DataAnalystTelemetry(BaseTelemetry):
    """
    Data Analyst specific telemetry manager.

    Extends BaseTelemetry with application-specific functionality
    and auto-instrumentation for FastAPI and requests library.
    Implements singleton pattern to ensure only one instance exists.
    """

    _instance: DataAnalystTelemetry | None = None
    _auto_instrumentation_setup: bool = False

    def __new__(cls) -> DataAnalystTelemetry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # type: ignore[assignment]
        return cls._instance  # type: ignore[return-value]

    def __init__(self) -> None:
        # Only initialize once (BaseTelemetry already handles this)
        super().__init__()

        # Setup auto-instrumentation on first init
        if not self._auto_instrumentation_setup:
            self._setup_auto_instrumentation()
            self._auto_instrumentation_setup = True

    def _setup_auto_instrumentation(self) -> None:
        """
        Setup auto-instrumentation for common libraries.

        Automatically instruments:
        - requests library (used by DataRobot client for API calls)
        - FastAPI (must be called separately with instrument_fastapi_app)
        """
        if RequestsInstrumentor is None:
            logger.warning(
                "RequestsInstrumentor not available. "
                "Install with: pip install opentelemetry-instrumentation-requests"
            )
            return

        try:
            RequestsInstrumentor().instrument()
            logger.info("Auto-instrumentation enabled for requests library")
        except Exception as e:
            logger.warning(f"Failed to setup auto-instrumentation: {e}")

    def instrument_fastapi_app(self, app: FastAPI) -> None:
        """
        Instrument a FastAPI application for automatic tracing.

        This should be called after creating your FastAPI app instance.

        Args:
            app: The FastAPI application instance to instrument

        Example:
            telemetry = DataAnalystTelemetry()
            app = FastAPI()
            telemetry.instrument_fastapi_app(app)
        """
        if FastAPIInstrumentor is None:
            logger.warning(
                "FastAPIInstrumentor not available. "
                "Install with: pip install opentelemetry-instrumentation-fastapi"
            )
            return

        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("Auto-instrumentation enabled for FastAPI application")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI app: {e}")

    def __enter__(self) -> DataAnalystTelemetry:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - gracefully shutdown telemetry."""
        self.shutdown()


telemetry = DataAnalystTelemetry()
