"""
OpenTelemetry instrumentation setup for monitoring API calls and performance metrics.
"""
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.http import HTTPClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.config import CONFIG

# Global reference to the Prometheus metric reader
_prometheus_metric_reader: Optional[PrometheusMetricReader] = None


def setup_telemetry(app, service_name: Optional[str] = None):
    """
    Set up OpenTelemetry instrumentation for the FastAPI application.

    Args:
        app: FastAPI application instance
        service_name: Optional service name override
    """
    global _prometheus_metric_reader

    if not CONFIG.OTEL_ENABLED:
        return

    service_name = service_name or CONFIG.OTEL_SERVICE_NAME

    # Create resource with service name
    resource = Resource.create({"service.name": service_name})

    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Add console exporter for traces (useful for debugging)
    if os.getenv("OTEL_LOG_LEVEL", "").lower() == "debug":
        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        tracer_provider.add_span_processor(span_processor)

    # Set up OTLP exporter if endpoint is configured
    if CONFIG.OTEL_EXPORTER_OTLP_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_exporter = OTLPSpanExporter(endpoint=CONFIG.OTEL_EXPORTER_OTLP_ENDPOINT)
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
        except ImportError:
            print("Warning: OTLP exporter not available. Install opentelemetry-exporter-otlp for OTLP support.")

    # Set up metrics with Prometheus exporter
    _prometheus_metric_reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[_prometheus_metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument HTTP client (for any outgoing HTTP requests)
    HTTPClientInstrumentor().instrument()

    print(f"OpenTelemetry instrumentation enabled for service: {service_name}")


def get_metrics_reader() -> Optional[PrometheusMetricReader]:
    """
    Get the Prometheus metric reader for exposing metrics endpoint.

    Returns:
        PrometheusMetricReader instance if telemetry is enabled, None otherwise
    """
    if not CONFIG.OTEL_ENABLED:
        return None
    return _prometheus_metric_reader

