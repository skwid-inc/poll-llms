import functools
import json
import logging
import os
import threading
import time
import traceback
import uuid
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Optional
from urllib.parse import urlparse

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.status import Status, StatusCode

from app_config import AppConfig
from secret_manager import access_secret

from urllib.parse import parse_qsl
logger = logging.getLogger(__name__)
# Global metric instruments
_metrics_instruments = {}


def singleton(func):
    """Decorator to ensure a function is only executed once."""
    func._lock = threading.Lock()
    func._initialized = False
    func._result = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not func._initialized:
            with func._lock:
                if not func._initialized:
                    func._result = func(*args, **kwargs)
                    func._initialized = True
        return func._result

    return wrapper


def get_env():
    return "prod"
    if (
        os.getenv("PAYMENT_API", "").lower() == "prod"
        or os.getenv("ENV", "").lower() == "prod"
    ):
        return "prod"
    return "dev"


def get_service_name():
    service_name = os.getenv("SERVICE_NAME")
    if service_name:
        return service_name
    service = AppConfig().service_name
    if service and service != "unknown":
        return f"{service}-{get_env()}"
    if (
        os.getenv("IS_INBOUND_CALL", "False") == "True"
        and os.getenv("USING_WATERFIELD") != "True"
    ):
        return f"salient-inbound-{get_env()}"
    return f"salient-outbound-{get_env()}"


def get_domain_name(url):
    try:
        return urlparse(url).netloc
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return url


def get_masked_url(url):
    try:
        domain = urlparse(url).netloc
        path = urlparse(url).path
        path_list = path.split("/")
        for i in range(len(path_list)):
            if path_list[i].isdigit():
                path_list[i] = "***"
        path = "/".join(path_list)
        return f"{domain}{path}"
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return url


def get_masked_response(response_text):
    """Mask sensitive information in response text and extract important information.

    Args:
        response_text (str): The response text to mask

    Returns:
        str: Masked response text or extracted information
    """
    try:
        # Handle JSON responses
        if isinstance(response_text, str) and response_text.strip().startswith("{"):
            data = json.loads(response_text)
            if "timestamp" in data:
                data["timestamp"] = "xxx"
            return json.dumps(data)

        # Handle HTML responses
        if isinstance(response_text, str) and "<title>" in response_text:
            import re

            # Extract title content
            title_match = re.search(r"<title>([^<]+)</title>", response_text)
            if title_match:
                return title_match.group(1).strip()

        # Default truncation for other responses
        MAX_LENGTH = 512
        if len(response_text) > MAX_LENGTH:
            return response_text[:MAX_LENGTH] + "..."

        return response_text
    except:
        # Fallback for any errors
        return response_text


def get_resource():
    return Resource(
        attributes={
            "repo": "taylor-fresh",
            "ENV": get_env(),
            SERVICE_NAME: get_service_name(),
            "inbound": os.getenv("IS_INBOUND_CALL", "False"),
            "hostname": os.getenv("HOSTNAME", ""),
            "fabric": os.getenv("FABRIC", ""),
        }
    )


def enrich_supabase_span(span, url: str, http_method: str) -> None:
    """
    Add comprehensive Supabase database attributes to a span.
    Handles table extraction, operation detection, and RPC calls.
    """
    if not span or not span.is_recording():
        return
    
    # Only enrich Supabase URLs
    if "supabase.co" not in url and "supabase.io" not in url:
        return

    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    query_items = dict(parse_qsl(parsed.query))

    # Always set db.system
    span.set_attribute("db.system", "supabase")

    # Handle RPC calls
    if "rpc" in path_parts:
        idx = path_parts.index("rpc")
        if idx + 1 < len(path_parts):
            procedure = path_parts[idx + 1]
            span.set_attribute("db.procedure", procedure)
        span.set_attribute("db.operation", "RPC")
        return

    # Extract table name from /rest/v1/<table>
    try:
        v1_idx = path_parts.index("v1")
        if v1_idx + 1 < len(path_parts):
            table = path_parts[v1_idx + 1]
            span.set_attribute("db.table", table)
    except ValueError:
        pass

    # Determine operation - prefer query params, fall back to HTTP
    if "select" in query_items:
        span.set_attribute("db.operation", "SELECT")
    else:
        # Handle both string and bytes methods
        if isinstance(http_method, bytes):
            http_method = http_method.decode('utf-8')
        
        verb_to_op = {
            "GET": "SELECT",
            "POST": "INSERT",
            "PUT": "UPDATE", 
            "PATCH": "UPDATE",
            "DELETE": "DELETE",
        }
        
        operation = verb_to_op.get(http_method.upper())
        if operation:
            span.set_attribute("db.operation", operation)


@singleton
def init_otel_logging():
    """Initialize OpenTelemetry logging with batched export to SigNoz."""
    resource = get_resource()

    # Create a logger provider
    logger_provider = LoggerProvider(resource=resource)

    # Create an OTLP exporter for logs
    otlp_exporter = OTLPLogExporter(
        endpoint="https://ingest.us.signoz.cloud:443",
        headers={"signoz-access-token": access_secret("signoz-key")},
    )

    # Add a batched processor to the logger provider with memory-friendly settings
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            otlp_exporter,
            max_export_batch_size=512,  # Number of logs to batch before sending
            schedule_delay_millis=5000,  # Send every 5 seconds even if not full
            max_queue_size=2048,  # Maximum logs to buffer in memory
        )
    )

    # Set the created logger provider
    set_logger_provider(logger_provider)

    # Return the OpenTelemetry logging handler to be added to Python's logger
    return LoggingHandler(logger_provider=logger_provider)


def get_skip_urls() -> list[str]:
    json_str = os.getenv("SKIP_ERROR_SPAN_URLS", "[]")
    return json.loads(json_str)


@singleton
def init_otel():
    """Initialize OpenTelemetry with thread-safe singleton pattern."""
    os.environ["OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE"] = "DELTA"
    resource = get_resource()
    traceProvider = TracerProvider(resource=resource)

    traceProvider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="https://ingest.us.signoz.cloud:443",
                headers={"signoz-access-token": access_secret("signoz-key")},
            )
        )
    )

    if get_env() == "dev":
        # Print spans to console to debug tracing issues
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        traceProvider.add_span_processor(console_processor)

    trace.set_tracer_provider(traceProvider)

    # Configure metric reader only in prod
    if get_env() == "prod":
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint="https://ingest.us.signoz.cloud:443",
                headers={"signoz-access-token": access_secret("signoz-key")},
            )
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
    else:
        # In non-prod, use metrics without OTLP exporter
        provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(provider)

    # `request_obj` is an instance of requests.PreparedRequest
    def request_hook(span, request_obj):
        span.set_attribute("http.domain", get_domain_name(request_obj.url))
        http_masked_url = get_masked_url(request_obj.url)
        span.set_attribute("http.masked_url", http_masked_url)

    # `request_obj` is an instance of requests.PreparedRequest
    # `response` is an instance of requests.Response
    def response_hook(span, request_obj, response):
        if response.status_code >= 400:
            http_masked_url = get_masked_url(request_obj.url)
            if http_masked_url not in get_skip_urls():
                truncated_response = get_masked_response(response.text)
                set_span_error(
                    span,
                    Exception(f"HTTP {response.status_code}: {truncated_response}"),
                )
                span.set_attribute("http.response.text", response.text)
            logger.error(
                f"HTTP response for {request_obj.url}: {response.status_code}: {response.text}"
            )

    RequestsInstrumentor().instrument(
        request_hook=request_hook, response_hook=response_hook
    )


    def httpx_request_hook(span, request_info):
        """Hook to add custom attributes to httpx spans."""
        if get_env() == "dev":
            print(f"HTTPX Request Hook: {request_info.method} {request_info.url}")
        
        url = str(request_info.url)
        span.set_attribute("http.domain", get_domain_name(url))
        http_masked_url = get_masked_url(url)
        span.set_attribute("http.masked_url", http_masked_url)
        
        enrich_supabase_span(span, url, request_info.method)


    def httpx_response_hook(span, request_info, response_info):
        """Hook to handle httpx response errors."""
        if get_env() == "dev":
            print(f"HTTPX Response Hook: {request_info.method} {request_info.url} -> {response_info.status_code}")
        
        if hasattr(response_info, 'status_code') and response_info.status_code >= 400:
            url = str(request_info.url)
            http_masked_url = get_masked_url(url)
            if http_masked_url not in get_skip_urls():
                error_msg = f"HTTP {response_info.status_code}"
                if hasattr(response_info, 'text'):
                    truncated_response = get_masked_response(response_info.text)
                    error_msg += f": {truncated_response}"
                set_span_error(span, Exception(error_msg))
    
    if not hasattr(HTTPXClientInstrumentor, '_instrumented'):
        HTTPXClientInstrumentor().instrument(
            request_hook=httpx_request_hook,
            response_hook=httpx_response_hook
        )
        HTTPXClientInstrumentor._instrumented = True
        if get_env() == "dev":
            print("HTTPX instrumentation applied with custom hooks")
    else:
        if get_env() == "dev":
            print("HTTPX already instrumented, skipping")

# Export a singleton tracer instance - must be after init_otel() to use configured provider
def get_tracer():
    """Get the tracer instance, ensuring OTEL is initialized."""
    init_otel()  # Ensure OTEL is initialized
    return trace.get_tracer(__name__)

tracer = get_tracer()


def get_meter_instrument(meter, instrument_type, name, description=None):
    """
    Get a metric instrument, creating it if it doesn't exist yet.

    Args:
        meter: The OpenTelemetry meter
        instrument_type: 'counter', 'histogram', or 'gauge'
        name: Name of the metric instrument
        description: Description of the metric instrument

    Returns:
        The metric instrument
    """
    key = f"{instrument_type}:{name}"
    if key not in _metrics_instruments:
        if instrument_type == "counter":
            _metrics_instruments[key] = meter.create_counter(
                name, description=description
            )
        elif instrument_type == "histogram":
            _metrics_instruments[key] = meter.create_histogram(
                name, description=description
            )
        elif instrument_type == "gauge":
            _metrics_instruments[key] = meter.create_gauge(
                name, description=description
            )

    return _metrics_instruments[key]


def measure_span(span_name="", start_time=None, status=StatusCode.OK, attributes=None):
    print(f"Measuring span: {span_name}")
    if attributes and "call_id" not in attributes:
        attributes["call_id"] = AppConfig().call_metadata.get("call_id", "unknown")
    if attributes and "campaign_id" not in attributes:
        attributes["campaign_id"] = AppConfig().call_metadata.get(
            "campaign_id", "unknown"
        )
    with tracer.start_as_current_span(
        span_name,
        start_time=int(start_time * 1000000000),
    ) as span:
        if attributes:
            span.set_attributes(attributes)
        span.set_status(Status(status))

def record_explicit_span(start_time, end_time, span_name="", attributes={}):
    span = tracer.start_span(
            span_name,
            start_time=int(start_time * 1e9),
        )
    span.set_attributes({**attributes})
    span.end(end_time=int(end_time * 1e9))
    logger.info("Recorded explicit span: %s and duration %.6f", span_name, end_time - start_time)

# Common attributes for all metrics
common_attrs = {
    "repo": "taylor-fresh",
    "ENV": get_env(),
    "service": get_service_name(),
}

def report_metrics(
    name="",
    instrument_type="counter",
    value=0,
    description: str = "",
    attributes: Optional[dict] = None,
):
    init_otel()
    meter = metrics.get_meter(__name__)
    attr = {}
    if attributes:
        attr = {**common_attrs, **attributes}
    else:
        attr = common_attrs.copy()
    instrument = get_meter_instrument(meter, instrument_type, name, description)
    if instrument_type == "counter":
        instrument.add(value, attr)
    elif instrument_type == "histogram":
        instrument.record(value, attr)
    elif instrument_type == "gauge":
        instrument.set(value, attr)


def report_gauge(
    name="", value=0, description: str = "", attributes: Optional[dict] = None
):
    report_metrics(name, "gauge", value, description, attributes)


def inc_counter(
    name="", value=1, description: str = "", attributes: Optional[dict] = None
):
    report_metrics(name, "counter", value, description, attributes)


def set_span_error(span, error: Exception):
    """Helper function to set error status and attributes on a span.

    Args:
        span: The OpenTelemetry span to set error on
        error: The exception that occurred
    """
    span.set_status(Status(StatusCode.ERROR))
    span.set_attribute(
        "error.type",
        type(error).__name__ if type(error).__name__ else "Unknown",
    )
    span.set_attribute("error.message", str(error))
    span.set_attribute(
        "error.stack_trace",
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    )
    if AppConfig().get_call_metadata():
        span.set_attribute(
            "call_id", AppConfig().get_call_metadata().get("call_id", "")
        )
        span.set_attribute(
            "account_number",
            AppConfig().get_call_metadata().get("account_number", ""),
        )
        span.set_attribute(
            "campaign_id",
            AppConfig().get_call_metadata().get("campaign_id", ""),
        )
        span.set_attribute("client_name", AppConfig().client_name)