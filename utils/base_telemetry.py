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

import functools
import inspect
import logging
import os
import time
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Optional,
    no_type_check,
    overload,
)

from opentelemetry import context, metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Note: LoggingInstrumentor not needed for basic telemetry setup
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExponentialBucketHistogramAggregation
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from typing_extensions import ParamSpec, Self, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


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
        
        # Telemetry enabled by default, disabled in local dev (start scripts set DISABLE_TELEMETRY=true)
        self.telemetry_enabled = os.environ.get("DISABLE_TELEMETRY") != "true"

        self._logger_provider: Optional[LoggerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._configured: bool = False
        self._startup_logged: bool = False  # Track if startup has been logged

        self._initialized = True

    def configure_logging(self) -> LoggerProvider:
        """
        Configure OpenTelemetry logging based on DataRobot patterns.
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
        otlp_exporter = OTLPMetricExporter(
            preferred_aggregation={Histogram: ExponentialBucketHistogramAggregation()},
        )

        # Create metric reader
        reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=1000,
        )

        # Create resource
        resource = Resource.create(
            {
                "service.name": f"{self.entity_type}-{self.entity_id}",
                "datarobot.service.priority": "p1",
            }
        )

        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                reader,
            ],
        )
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
        # Skip OpenTelemetry handler if telemetry is disabled
        if not self.telemetry_enabled:
            return logging.getLogger(name)
        
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

    def get_meter(self, name: str) -> metrics.Meter:
        """
        Get a meter instance for the given name using OpenTelemetry global API.
        """
        if self.telemetry_enabled and not self._meter_provider:
            self.configure_metrics()
        return metrics.get_meter(name)

    def get_tracer(self, name: str) -> trace.Tracer:
        """
        Get a tracer instance for the given name using OpenTelemetry global API.
        """
        if self.telemetry_enabled and not self._tracer_provider:
            self.configure_tracing()
        return trace.get_tracer(name)

    def get_context(self) -> context.Context:
        """
        Returns current OTEL context. To cross thread boundaries, you'll need to do
        get_context in spawning thread and set_context in spawned thread.
        """
        return context.get_current()

    def set_context(self, otel_context: context.Context) -> Any:
        """Sets OTEL context."""
        return context.attach(otel_context)

    def reset_context(self, token: Any) -> None:
        context.detach(token)

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

    @overload
    def trace(
        self: Self,
        func: Callable[P, Coroutine[T, None, None]],
    ) -> Callable[P, Coroutine[T, None, None]]: ...

    @overload
    def trace(
        self: Self,
        func: Callable[P, AsyncGenerator[T, None]],
    ) -> Callable[P, AsyncGenerator[T, None]]: ...

    @overload
    def trace(
        self: Self,
        func: Callable[P, Generator[T, None, None]],
    ) -> Callable[P, Generator[T, None, None]]: ...

    @overload
    def trace(self: Self, func: Callable[P, T]) -> Callable[P, T]: ...

    @no_type_check
    def trace(self: Self, func: Any) -> Any:
        """
        Wrap the execution of the decorated function in an OTEL span sharing the same name as the function.
        WARNING: There are sharp edges with this decorator if applied to functions that are reflected on.
        (I've seen this with methods in utils.rest_api.)
        """
        tracer = self.get_tracer("talk-to-my-data-application-tracer")

        span_name = f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_inner(*args, **kwargs):
                with tracer.start_as_current_span(span_name):
                    return await func(*args, **kwargs)

            return async_inner
        elif inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def inner_asyncgen(*args, **kwargs):
                with tracer.start_as_current_span(span_name):
                    async for x in func(*args, **kwargs):
                        yield x

            return inner_asyncgen
        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def inner_gen(*args, **kwargs):
                with tracer.start_as_current_span(span_name):
                    for x in func(*args, **kwargs):
                        yield x

            return inner_gen
        elif inspect.isfunction(func):

            @functools.wraps(func)
            def inner(*args, **kwargs):
                with tracer.start_as_current_span(span_name):
                    return func(*args, **kwargs)

            return inner
        else:
            raise ValueError(
                f"instrument can only decorate a function type, while {span_name} is a {type(func)}."
            )

    @functools.cache
    def _function_histogram(self: Self, name: str) -> metrics.Histogram:
        meter = self.get_meter("talk-to-my-data-application-meter")
        return meter.create_histogram(
            f"ttmdata.function.{name}", "s", "A histogram recording function timings."
        )

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        start_time = time.time_ns()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            end_time = time.time_ns()
            histogram = self._function_histogram(name)
            histogram.record((end_time - start_time) / 1e9, {"success": success})

    @overload
    def meter(
        self: Self,
        func: Callable[P, Coroutine[T, None, None]],
    ) -> Callable[P, Coroutine[T, None, None]]: ...

    @overload
    def meter(
        self: Self,
        func: Callable[P, AsyncGenerator[T, None]],
    ) -> Callable[P, AsyncGenerator[T, None]]: ...

    @overload
    def meter(
        self: Self,
        func: Callable[P, Generator[T, None, None]],
    ) -> Callable[P, Generator[T, None, None]]: ...

    @overload
    def meter(self: Self, func: Callable[P, T]) -> Callable[P, T]: ...

    @no_type_check
    def meter(self: Self, func: Any) -> Any:
        """
        Wrap the execution of the decorated function in an OTEL span sharing the same name as the function.
        WARNING: There are sharp edges with this decorator if applied to functions that are reflected on.
        (I've seen this with methods in utils.rest_api.)
        """
        span_name = f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_inner(*args, **kwargs):
                with self.time(span_name):
                    return await func(*args, **kwargs)

            return async_inner
        elif inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def inner_asyncgen(*args, **kwargs):
                with self.time(span_name):
                    async for x in func(*args, **kwargs):
                        yield x

            return inner_asyncgen
        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def inner_gen(*args, **kwargs):
                with self.time(span_name):
                    for x in func(*args, **kwargs):
                        yield x

            return inner_gen
        elif inspect.isfunction(func):

            @functools.wraps(func)
            def inner(*args, **kwargs):
                with self.time(span_name):
                    return func(*args, **kwargs)

            return inner
        else:
            raise ValueError(
                f"instrument can only decorate a function type, while {span_name} is a {type(func)}."
            )

    @overload
    def meter_and_trace(
        self: Self,
        func: Callable[P, Coroutine[T, None, None]],
    ) -> Callable[P, Coroutine[T, None, None]]: ...

    @overload
    def meter_and_trace(
        self: Self,
        func: Callable[P, AsyncGenerator[T, None]],
    ) -> Callable[P, AsyncGenerator[T, None]]: ...

    @overload
    def meter_and_trace(
        self: Self,
        func: Callable[P, Generator[T, None, None]],
    ) -> Callable[P, Generator[T, None, None]]: ...

    @overload
    def meter_and_trace(self: Self, func: Callable[P, T]) -> Callable[P, T]: ...

    @no_type_check
    def meter_and_trace(self: Self, func: Any) -> Any:
        return functools.wraps(func)(self.meter(self.trace(func)))
