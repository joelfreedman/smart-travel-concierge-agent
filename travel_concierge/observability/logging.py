"""Structured JSON logging with automatic PII scrubbing and trace correlation.

Replaces unstructured stdout printing with machine-parseable JSON records
suitable for Google Cloud Logging and local audit inspection, explicitly
supporting Intent-before-execution and Outcome-after-execution event schemas.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from travel_concierge.security.pii_redactor import PIIRedactor


class JsonLogFormatter(logging.Formatter):
    """Formats Python LogRecords as single-line Cloud Logging compatible JSON."""

    def __init__(self, redactor: PIIRedactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor or PIIRedactor()

    def format(self, record: logging.LogRecord) -> str:
        span_ctx = trace.get_current_span().get_span_context()
        trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
        span_id = format(span_ctx.span_id, "016x") if span_ctx.is_valid else None

        scrubbed_message = self._redactor.redact_text(record.getMessage())
        structured_payload = getattr(record, "structured_payload", None)
        redacted_categories = list(scrubbed_message.redacted_categories)

        safe_payload: Any = None
        if structured_payload is not None:
            safe_payload, payload_pii = self._redactor.redact_mapping(structured_payload)
            redacted_categories = sorted(set(redacted_categories) | set(payload_pii))

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "event_type": getattr(record, "event_type", "agent.log"),
            "message": scrubbed_message.redacted_text,
            "trace_id": trace_id,
            "span_id": span_id,
            "pii_redacted_categories": redacted_categories,
        }
        if safe_payload is not None:
            log_entry["payload"] = safe_payload
        if record.exc_info:
            # Avoid leaking raw PII in exception messages by scrubbing formatted info.
            exc_text = self.formatException(record.exc_info)
            log_entry["exception"] = self._redactor.redact_text(exc_text).redacted_text

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class StructuredAgentLogger:
    """High-level structured logger capturing Intent vs. Outcome and security events."""

    def __init__(
        self,
        name: str = "travel_concierge",
        *,
        level: str = "INFO",
        redactor: PIIRedactor | None = None,
    ) -> None:
        self._redactor = redactor or PIIRedactor()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonLogFormatter(redactor=self._redactor))
            self._logger.addHandler(handler)

    @property
    def raw_logger(self) -> logging.Logger:
        """Expose underlying stdlib Logger for attaching test handlers."""
        return self._logger

    def emit(
        self,
        event_type: str,
        message: str,
        *,
        level: int = logging.INFO,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit a PII-scrubbed structured JSON log record."""
        self._logger.log(
            level,
            message,
            extra={
                "event_type": event_type,
                "structured_payload": payload or {},
            },
        )

    def log_intent(
        self,
        *,
        actor: str,
        action: str,
        session_id: str | None = None,
        invocation_id: str | None = None,
        intended_args: dict[str, Any] | None = None,
        rationale: str | None = None,
    ) -> None:
        """Record an agent's *intended* action immediately BEFORE execution."""
        self.emit(
            "agent.intent",
            f"Intent recorded: {actor} intends to execute {action}",
            payload={
                "phase": "intent",
                "actor": actor,
                "action": action,
                "session_id": session_id,
                "invocation_id": invocation_id,
                "intended_args": intended_args or {},
                "rationale": rationale or "agent_planned_step",
            },
        )

    def log_outcome(
        self,
        *,
        actor: str,
        action: str,
        outcome_status: str,
        latency_ms: float | None = None,
        session_id: str | None = None,
        invocation_id: str | None = None,
        outcome_summary: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        """Record the *actual* outcome immediately AFTER execution completes."""
        level = logging.WARNING if outcome_status in {"error", "blocked"} else logging.INFO
        self.emit(
            "agent.outcome",
            f"Outcome recorded: {actor} completed {action} with status={outcome_status}",
            level=level,
            payload={
                "phase": "outcome",
                "actor": actor,
                "action": action,
                "outcome_status": outcome_status,
                "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
                "session_id": session_id,
                "invocation_id": invocation_id,
                "error_code": error_code,
                "outcome_summary": outcome_summary or {},
            },
        )

    def log_security_event(
        self,
        *,
        rule: str,
        action_taken: str,
        severity: str = "WARNING",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record guardrail triggers, PII redactions, or HITL enforcement blocks."""
        level = getattr(logging, severity.upper(), logging.WARNING)
        self.emit(
            "agent.security",
            f"Security guardrail '{rule}' -> {action_taken}",
            level=level,
            payload={
                "rule": rule,
                "action_taken": action_taken,
                "details": details or {},
            },
        )
