"""OpenTelemetry distributed tracing for end-to-end ADK request correlation.

Links every user query -> root coordinator -> specialist sub-agents -> LLM calls
-> tool executions into a single distributed trace tree with scrubbed attributes
and explicit ``intent`` / ``outcome`` span events.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from travel_concierge.config.settings import AppSettings
from travel_concierge.security.pii_redactor import PIIRedactor


class TelemetryManager:
    """Configures OpenTelemetry tracing and manages scrubbed agent spans."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        exporter: SpanExporter | None = None,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self._settings = settings or AppSettings.from_env()
        self._redactor = redactor or PIIRedactor(settings=self._settings)
        resource = Resource.create(
            {
                "service.name": self._settings.otel_service_name,
                "deployment.environment": self._settings.environment,
            }
        )
        self._provider = TracerProvider(resource=resource)

        if exporter is not None:
            self._provider.add_span_processor(SimpleSpanProcessor(exporter))
        elif self._settings.otel_export_cloud_trace and self._settings.gcp_project_id:
            try:
                from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                cloud_exporter = CloudTraceSpanExporter(
                    project_id=self._settings.gcp_project_id
                )
                self._provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
            except ImportError:
                # Cloud Trace exporter is an optional production extra.
                pass

        self._tracer: Tracer = self._provider.get_tracer(
            "travel_concierge.telemetry", "0.1.0"
        )

    @property
    def provider(self) -> TracerProvider:
        """Return the configured OpenTelemetry TracerProvider."""
        return self._provider

    @property
    def tracer(self) -> Tracer:
        """Return the agent OpenTelemetry Tracer."""
        return self._tracer

    def _safe_attributes(self, attributes: dict[str, Any] | None) -> dict[str, Any]:
        """Scrub PII and coerce values to OpenTelemetry-compatible primitives."""
        if not attributes:
            return {}
        scrubbed, _ = self._redactor.redact_mapping(attributes)
        safe: dict[str, Any] = {}
        for key, value in scrubbed.items():
            if isinstance(value, (bool, int, float, str)):
                safe[key] = value
            elif value is None:
                continue
            else:
                safe[key] = str(value)
        return safe

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        parent_context: Any | None = None,
    ) -> Span:
        """Start an OpenTelemetry span with PII-scrubbed attributes."""
        return self._tracer.start_span(
            name,
            context=parent_context,
            attributes=self._safe_attributes(attributes),
        )

    def record_intent_event(self, span: Span, action: str, details: dict[str, Any]) -> None:
        """Attach a scrubbed ``agent.intent`` event to an active span."""
        span.add_event(
            "agent.intent",
            attributes=self._safe_attributes({"action": action, **details}),
        )

    def record_outcome_event(
        self,
        span: Span,
        action: str,
        status: str,
        details: dict[str, Any],
    ) -> None:
        """Attach a scrubbed ``agent.outcome`` event and set span status."""
        span.add_event(
            "agent.outcome",
            attributes=self._safe_attributes(
                {"action": action, "outcome.status": status, **details}
            ),
        )
        if status in {"error", "blocked"}:
            span.set_status(Status(StatusCode.ERROR, description=status))
        else:
            span.set_status(Status(StatusCode.OK))

    @contextmanager
    def traced_operation(
        self,
        span_name: str,
        *,
        action: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager creating a linked child span with intent/outcome capture."""
        with self._tracer.start_as_current_span(
            span_name,
            attributes=self._safe_attributes(attributes),
        ) as span:
            self.record_intent_event(span, action=action, details=attributes or {})
            try:
                yield span
            except Exception as exc:
                self.record_outcome_event(
                    span,
                    action=action,
                    status="error",
                    details={"error_type": type(exc).__name__},
                )
                raise

    def shutdown(self) -> None:
        """Flush and shut down the underlying OpenTelemetry TracerProvider."""
        self._provider.shutdown()
