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
Open Telemetry module for DataRobot Custom Applications.

This module provides a reusable telemetry foundation that can be extended
for specific Custom Applications while maintaining consistent datavolt patterns.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import os
import time
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
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

from core.telemetry.logging import RedactingFormatter

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

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:
    HTTPXClientInstrumentor = None  # type: ignore[assignment, misc]

try:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
except ImportError:
    LoggingInstrumentor = None  # type: ignore[assignment, misc]

try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
except ImportError:
    SQLAlchemyInstrumentor = None  # type: ignore[assignment, misc]

P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_EXCLUDED_TRACE_SPAN_NAMES = frozenset({"app.middleware._initialize_session"})

_otel_handler_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_otel_handler_active", default=False
)


class _SafeLoggingHandler(LoggingHandler):
    """LoggingHandler with ContextVar recursion guard (backport of opentelemetry-python-contrib #4302)."""

    def emit(self, record: logging.LogRecord) -> None:
        if _otel_handler_active.get():
            return
        token = _otel_handler_active.set(True)
        try:
            super().emit(record)
        finally:
            _otel_handler_active.reset(token)


class OTLPConnectionErrorFilter(logging.Filter):
    """
    Filter to suppress connection errors from urllib3/requests when OTLP collector is unavailable.

    This prevents error spam in Chronosphere when the OTLP collector endpoint isn't properly
    configured or available (e.g., localhost:4318 connection refused errors).
    """

    def __init__(self, warning_callback: Optional[Callable[[], None]] = None):
        super().__init__()
        self.warning_callback = warning_callback

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress the log record, True to allow it."""
        should_suppress = False

        # Suppress urllib3 connection errors related to OTLP endpoints
        if record.name.startswith("urllib3.connectionpool"):
            message = record.getMessage()
            if "HTTPConnectionPool" in message and (
                ":4318" in message  # Default OTLP port
                or "/v1/metrics" in message
                or "/v1/traces" in message
                or "/v1/logs" in message
            ):
                should_suppress = True

        # Suppress requests connection errors related to OTLP
        if record.name.startswith("requests."):
            message = record.getMessage()
            if "ConnectionError" in message and ":4318" in message:
                should_suppress = True

        if should_suppress:
            if self.warning_callback:
                self.warning_callback()
            return False

        return True


class OTel:
    """
    Open Telemetry manager for DataRobot Custom Applications.

    Provides OpenTelemetry configuration following datavolt patterns.
    Implements singleton pattern to ensure only one instance exists per process.
    """

    _instance: Optional[OTel] = None
    _initialized: bool = False
    _auto_instrumentation_setup: bool = False

    def __new__(
        cls, entity_type: str = "custom_application", entity_id: Optional[str] = None
    ) -> OTel:
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

        # Auto-disable telemetry if OTLP endpoint is not configured
        if self.telemetry_enabled and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            # Check if internal endpoint is set (fallback)
            if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT_INTERNAL"):
                self.telemetry_enabled = False
                logging.getLogger(__name__).warning(
                    "OTEL_EXPORTER_OTLP_ENDPOINT not set. Disabling telemetry to prevent connection errors."
                )

        self._logger_provider: Optional[LoggerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._resource: Optional[Resource] = None
        self._configured: bool = False
        self._startup_logged: bool = False  # Track if startup has been logged
        self._otlp_warning_logged: bool = (
            False  # Track if OTLP connection warning has been logged
        )

        # Install filter to suppress OTLP connection errors from spamming logs
        # We keep this even if telemetry is disabled, just in case something tries to force it
        self._install_otlp_error_filter()

        # Setup auto-instrumentation on first init
        if not self._auto_instrumentation_setup:
            self._setup_auto_instrumentation()
            self._auto_instrumentation_setup = True

        self._initialized = True

    def _get_resource(self) -> Resource:
        if self._resource is None:
            attrs: dict[str, str] = {
                "datarobot.service.priority": "p1",
            }
            # Only set service.name if the platform hasn't already provided OTEL_SERVICE_NAME.
            # Resource.create() merges env vars at lower precedence than explicit attrs, so
            # setting it here would shadow the platform value if they ever diverge.
            if not os.environ.get("OTEL_SERVICE_NAME"):
                attrs["service.name"] = f"{self.entity_type}-{self.entity_id}"
            if self.entity_id:
                attrs["datarobot.application.id"] = self.entity_id
            pod_name = os.environ.get("HOSTNAME")
            if pod_name:
                attrs["k8s.pod.name"] = pod_name
            self._resource = Resource.create(attrs)
        return self._resource

    def _install_otlp_error_filter(self) -> None:
        """Install logging filter to suppress OTLP connection errors."""
        otlp_filter = OTLPConnectionErrorFilter(self._log_otlp_warning)

        # Apply to urllib3 logger
        urllib3_logger = logging.getLogger("urllib3.connectionpool")
        urllib3_logger.addFilter(otlp_filter)

        # Apply to requests logger
        requests_logger = logging.getLogger("requests")
        requests_logger.addFilter(otlp_filter)

    def _log_otlp_warning(self) -> None:
        """Log a warning about OTLP connection failure (only once)."""
        if not self._otlp_warning_logged:
            self._otlp_warning_logged = True
            # Use a logger that is NOT filtered or ensure this message doesn't trigger the filter
            logger = logging.getLogger(__name__)
            logger.warning(
                "OTLP collector connection failed. Telemetry data may be lost. "
                "Suppressing further connection errors to prevent log spam. "
                "Check OTEL_EXPORTER_OTLP_ENDPOINT configuration."
            )

    def _get_excluded_trace_span_names(self) -> frozenset[str]:
        configured_span_names = {
            span_name.strip()
            for span_name in os.environ.get("OTEL_EXCLUDED_TRACE_SPAN_NAMES", "").split(
                ","
            )
            if span_name.strip()
        }
        return DEFAULT_EXCLUDED_TRACE_SPAN_NAMES | configured_span_names

    def _is_trace_span_excluded(self, span_name: str) -> bool:
        return span_name in self._get_excluded_trace_span_names()

    def _setup_auto_instrumentation(self) -> None:
        """
        Setup auto-instrumentation for common libraries.

        Automatically instruments:
        - requests library (used by DataRobot client for API calls)
        - httpx library (if installed)
        - FastAPI (must be called separately with instrument_fastapi_app)
        """
        if RequestsInstrumentor is not None:
            try:
                RequestsInstrumentor().instrument()
                logging.getLogger(__name__).info(
                    "Auto-instrumentation enabled for requests library"
                )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to setup requests auto-instrumentation: {e}"
                )
        else:
            logging.getLogger(__name__).warning(
                "RequestsInstrumentor not available. "
                "Install with: pip install opentelemetry-instrumentation-requests"
            )

        if HTTPXClientInstrumentor is not None:
            try:
                HTTPXClientInstrumentor().instrument()
                logging.getLogger(__name__).info(
                    "Auto-instrumentation enabled for httpx library"
                )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to setup httpx auto-instrumentation: {e}"
                )

        if SQLAlchemyInstrumentor is not None:
            try:
                SQLAlchemyInstrumentor().instrument()
                logging.getLogger(__name__).info(
                    "Auto-instrumentation enabled for SQLAlchemy"
                )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to setup SQLAlchemy auto-instrumentation: {e}"
                )

    def instrument_fastapi_app(self, app: FastAPI) -> None:
        """
        Instrument a FastAPI application for automatic tracing.

        This should be called after creating your FastAPI app instance.

        Args:
            app: The FastAPI application instance to instrument

        Example:
            otel = OTel()
            app = FastAPI()
            otel.instrument_fastapi_app(app)
        """
        if FastAPIInstrumentor is None:
            logging.getLogger(__name__).warning(
                "FastAPIInstrumentor not available. "
                "Install with: pip install opentelemetry-instrumentation-fastapi"
            )
            return

        try:
            # Ensure tracer provider/exporter is configured before instrumenting FastAPI
            if self.telemetry_enabled and not self._tracer_provider:
                self.configure_tracing()

            FastAPIInstrumentor.instrument_app(
                app,
                # - //[^/]+/$ matches the root path also excludes kube-probe which has
                #  {full_path:path} route. (http://host:port/ with no further segments)
                # - /health$ matches the health endpoint
                # - /assets/.* matches static asset paths
                excluded_urls=r"//[^/]+/$,/health$,/assets/.*",
            )
            logging.getLogger(__name__).info(
                "Auto-instrumentation enabled for FastAPI application"
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to instrument FastAPI app: {e}"
            )

    def configure_logging(self) -> LoggerProvider:
        """
        Configure OpenTelemetry logging based on DataRobot patterns.
        """
        if self._logger_provider:
            return self._logger_provider

        # Create logger provider
        logger_provider = LoggerProvider(resource=self._get_resource())
        set_logger_provider(logger_provider)

        # Create OTLP exporter
        try:
            otlp_exporter = OTLPLogExporter()
            # Create batch processor
            batch_processor = BatchLogRecordProcessor(otlp_exporter)
            logger_provider.add_log_record_processor(batch_processor)
        except Exception as e:
            # Log warning but don't crash
            logging.getLogger(__name__).warning(
                f"Failed to initialize OTLP logging exporter: {e}"
            )

        # Inject trace_id/span_id into every stdlib log record so stdout formatters
        # include them automatically (picked up as extra fields by Json/TextFormatter).
        if LoggingInstrumentor is not None:
            LoggingInstrumentor().instrument()

        # Attach to root logger so every logger in the process exports via propagation,
        # regardless of how it was created (logging.getLogger() vs otel.get_logger()).
        root_handler = _SafeLoggingHandler(
            level=logging.INFO, logger_provider=logger_provider
        )
        root_handler.setFormatter(RedactingFormatter(logging.Formatter()))
        logging.getLogger().addHandler(root_handler)

        self._logger_provider = logger_provider
        return logger_provider

    def configure_metrics(self) -> MeterProvider:
        """
        Configure OpenTelemetry metrics based on datavolt patterns.
        """
        if self._meter_provider:
            return self._meter_provider

        # Create OTLP exporter
        try:
            otlp_exporter = OTLPMetricExporter(
                preferred_aggregation={
                    Histogram: ExponentialBucketHistogramAggregation()
                },
            )

            # Create metric reader
            reader = PeriodicExportingMetricReader(
                exporter=otlp_exporter,
                export_interval_millis=30000,
            )

            # Create meter provider
            meter_provider = MeterProvider(
                resource=self._get_resource(),
                metric_readers=[
                    reader,
                ],
            )
            metrics.set_meter_provider(meter_provider)
            self._meter_provider = meter_provider
            return meter_provider
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize OTLP metrics exporter: {e}"
            )
            meter_provider = MeterProvider(resource=self._get_resource())
            metrics.set_meter_provider(meter_provider)
            self._meter_provider = meter_provider
            return meter_provider

    def configure_tracing(self) -> TracerProvider:
        """
        Configure OpenTelemetry tracing based on datavolt patterns.
        """
        if self._tracer_provider:
            return self._tracer_provider

        # Create tracer provider
        tracer_provider = TracerProvider(resource=self._get_resource())
        trace.set_tracer_provider(tracer_provider)

        # Create OTLP exporter
        try:
            otlp_exporter = OTLPSpanExporter()

            # Create batch processor
            batch_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(batch_processor)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize OTLP tracing exporter: {e}"
            )

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
        if not self.telemetry_enabled:
            return logging.getLogger(name)

        if not self._logger_provider:
            self.configure_logging()

        logger = logging.getLogger(name)
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
        # prevent duplicate startup logs
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

    def __enter__(self) -> OTel:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - gracefully shutdown telemetry."""
        self.shutdown()

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
        span_name = f"{func.__module__}.{func.__qualname__}"

        if self._is_trace_span_excluded(span_name):
            return func

        tracer = self.get_tracer("application-tracer")

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
        meter = self.get_meter("application-meter")
        return meter.create_histogram(
            f"function.{name}", "s", "A histogram recording function timings."
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


otel = OTel()
