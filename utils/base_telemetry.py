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
Base telemetry module for DataRobot Custom Applications.

This module provides a reusable telemetry foundation that can be extended
for specific Custom Applications while maintaining consistent datavolt patterns.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Note: LoggingInstrumentor not needed for basic telemetry setup
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class BaseTelemetry:
    """
    Base telemetry manager for DataRobot Custom Applications.

    Provides OpenTelemetry configuration following datavolt patterns.
    Implements singleton pattern to ensure only one instance exists per process.

    This class should be extended for specific applications.
    """

    _instance: Optional[BaseTelemetry] = None
    _initialized: bool = False

    def __new__(
        cls, entity_type: str = "custom_application", entity_id: Optional[str] = None
    ) -> BaseTelemetry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, entity_type: str = "custom_application", entity_id: Optional[str] = None
    ):
        # Only initialize once
        if self._initialized:
            return

        self.entity_type = entity_type
        self.entity_id = entity_id or os.environ.get("APPLICATION_ID")

        self._logger_provider: Optional[LoggerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._configured: bool = False
        self._startup_logged: bool = False  # Track if startup has been logged

        self._initialized = True

    def configure_logging(self) -> LoggerProvider:
        """
        Configure OpenTelemetry logging based on datavolt patterns.
        """
        if self._logger_provider:
            return self._logger_provider

        # Create resource with application context
        resource = Resource.create(
            {
                "service.name": f"{self.entity_type}-{self.entity_id}",
                "datarobot.service.priority": "p1",
            }
        )

        # Create logger provider
        logger_provider = LoggerProvider(resource=resource)
        set_logger_provider(logger_provider)

        # Create OTLP exporter
        otlp_exporter = OTLPLogExporter()

        # Create batch processor
        batch_processor = BatchLogRecordProcessor(otlp_exporter)
        logger_provider.add_log_record_processor(batch_processor)

        # Note: LoggingHandler will be created per logger in get_logger() method
        self._logger_provider = logger_provider
        return logger_provider

    def configure_metrics(self) -> MeterProvider:
        """
        Configure OpenTelemetry metrics based on datavolt patterns.
        """
        if self._meter_provider:
            return self._meter_provider

        # Create OTLP exporter
        otlp_exporter = OTLPMetricExporter()

        # Create metric reader
        reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter, export_interval_millis=1000
        )

        # Create resource
        resource = Resource.create(
            {
                "service.name": f"{self.entity_type}-{self.entity_id}",
                "datarobot.service.priority": "p1",
            }
        )

        # Create meter provider
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

        self._meter_provider = meter_provider
        return meter_provider

    def configure_tracing(self) -> TracerProvider:
        """
        Configure OpenTelemetry tracing based on datavolt patterns.
        """
        if self._tracer_provider:
            return self._tracer_provider

        # Create resource
        resource = Resource.create(
            {
                "service.name": f"{self.entity_type}-{self.entity_id}",
                "datarobot.service.priority": "p1",
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter()

        # Create batch processor
        batch_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(batch_processor)

        self._tracer_provider = tracer_provider
        return tracer_provider

    def configure_all(self) -> None:
        """Configure all telemetry providers (logging, metrics, tracing)."""
        if self._configured:
            return

        self.configure_logging()
        self.configure_metrics()
        self.configure_tracing()

        # Note: Automatic instrumentation not needed for basic telemetry

        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a Python logger configured to send logs through OpenTelemetry.
        """
        if not self._logger_provider:
            self.configure_logging()

        # Create a standard Python logger
        logger = logging.getLogger(name)

        # Check if we already added the OpenTelemetry handler to avoid duplicates
        otel_handler_exists = any(
            isinstance(handler, LoggingHandler) for handler in logger.handlers
        )

        if not otel_handler_exists:
            # Create OpenTelemetry logging handler
            handler = LoggingHandler(
                level=logging.INFO, logger_provider=self._logger_provider
            )

            # Set a formatter for better log structure
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def get_meter(self, name: str) -> Any:
        """
        Get a meter instance for the given name using OpenTelemetry global API.
        """
        if not self._meter_provider:
            self.configure_metrics()
        return metrics.get_meter(name)

    def get_tracer(self, name: str) -> Any:
        """
        Get a tracer instance for the given name using OpenTelemetry global API.
        """
        if not self._tracer_provider:
            self.configure_tracing()
        return trace.get_tracer(name)

    def shutdown(self) -> None:
        """
        Gracefully shutdown all telemetry providers.
        """
        if self._logger_provider:
            self._logger_provider.shutdown()  # type: ignore[no-untyped-call]
        if self._meter_provider:
            self._meter_provider.shutdown()
        if self._tracer_provider:
            self._tracer_provider.shutdown()  # type: ignore[no-untyped-call]

        # Allow time for final exports (as seen in datavolt examples)
        time.sleep(1)

    def log_application_start(self, application_name: str = "Application") -> None:
        """
        Log application startup event (only once per process).

        Args:
            application_name: Name of the application for logging context
        """
        # Only log startup once per process to prevent Streamlit rerun spam
        if self._startup_logged:
            return

        self._startup_logged = True
        logger = self.get_logger(f"{self.entity_type}.startup")
        logger.info(
            f"{application_name} starting up",
            extra={
                "application_id": self.entity_id,
                "application_type": self.entity_type,
            },
        )
