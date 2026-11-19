"""Tests for JSON logging, Intent/Outcome capture, OTel tracing, routing, secrets, and API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.adk.agents import LlmAgent, SequentialAgent
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from travel_concierge.agent import create_travel_concierge_runtime
from travel_concierge.app_server import create_fastapi_app
from travel_concierge.config.secrets import SecretProvider, SecretResolutionError
from travel_concierge.config.settings import AppSettings, TaskComplexity
from travel_concierge.observability.logging import JsonLogFormatter


class _MemoryListHandler(logging.Handler):
    def __init__(self, formatter: logging.Formatter) -> None:
        super().__init__()
        self.setFormatter(formatter)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def _isolated_settings(tmp_path: Path) -> AppSettings:
    base = AppSettings.from_env()
    return AppSettings(
        **{
            **base.__dict__,
            "session_db_path": tmp_path / "obs_sessions.db",
            "memory_db_path": tmp_path / "obs_memory.db",
            "enable_cloud_dlp": False,
            "otel_export_cloud_trace": False,
        }
    )


def test_multi_agent_hierarchy_and_strategic_model_routing(tmp_path: Path) -> None:
    """Coordinator routes to Sequential pipeline (Flash -> Pro) and Booking specialist (Flash)."""
    settings = _isolated_settings(tmp_path)
    runtime = create_travel_concierge_runtime(settings=settings)

    assert settings.model_routing.select_model(TaskComplexity.FAST_ROUTING) == "gemini-2.5-flash"
    assert settings.model_routing.select_model(TaskComplexity.COMPLEX_PLANNING) == "gemini-2.5-pro"

    root = runtime.root_agent
    assert isinstance(root, LlmAgent)
    assert root.name == "travel_concierge_coordinator"
    assert root.model == "gemini-2.5-flash"
    assert len(root.sub_agents) == 2

    pipeline = root.sub_agents[0]
    booking = root.sub_agents[1]
    assert isinstance(pipeline, SequentialAgent)
    assert pipeline.name == "trip_planning_pipeline"
    assert pipeline.sub_agents[0].name == "trip_intake_specialist"
    assert pipeline.sub_agents[0].model == "gemini-2.5-flash"
    assert pipeline.sub_agents[1].name == "itinerary_budget_architect"
    assert pipeline.sub_agents[1].model == "gemini-2.5-pro"
    assert booking.name == "booking_execution_specialist"
    assert booking.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_intent_vs_outcome_json_logging_and_distributed_otel_spans(tmp_path: Path) -> None:
    """Verify parent-child OTel spans and scrubbed Intent + Outcome JSON log correlation."""
    exporter = InMemorySpanExporter()
    runtime = create_travel_concierge_runtime(
        settings=_isolated_settings(tmp_path), span_exporter=exporter
    )

    handler = _MemoryListHandler(JsonLogFormatter(redactor=runtime.redactor))
    runtime.logger.raw_logger.addHandler(handler)
    try:
        session = SimpleNamespace(id="sess_obs_1", state={})
        inv_ctx = SimpleNamespace(invocation_id="inv_obs_1", session=session, user_id="usr_1")
        cb_ctx = SimpleNamespace(
            invocation_id="inv_obs_1",
            agent_name="trip_intake_specialist",
            session=session,
            state=session.state,
        )
        tool = SimpleNamespace(name="search_budget_flights")
        tool_ctx = SimpleNamespace(
            invocation_id="inv_obs_1",
            agent_name="trip_intake_specialist",
            function_call_id="fc_01",
            session=session,
            state=session.state,
        )

        await runtime.guardrail_plugin.before_run_callback(invocation_context=inv_ctx)  # type: ignore[arg-type]
        await runtime.guardrail_plugin.before_agent_callback(
            agent=runtime.root_agent, callback_context=cb_ctx  # type: ignore[arg-type]
        )
        await runtime.guardrail_plugin.before_tool_callback(
            tool=tool,  # type: ignore[arg-type]
            tool_args={"origin_iata": "AUS", "note": "contact bob@example.com"},
            tool_context=tool_ctx,  # type: ignore[arg-type]
        )
        await runtime.guardrail_plugin.after_tool_callback(
            tool=tool,  # type: ignore[arg-type]
            tool_args={"origin_iata": "AUS"},
            tool_context=tool_ctx,  # type: ignore[arg-type]
            result={"status": "ok", "flights": []},
        )
        await runtime.guardrail_plugin.after_agent_callback(
            agent=runtime.root_agent, callback_context=cb_ctx  # type: ignore[arg-type]
        )
        await runtime.guardrail_plugin.after_run_callback(invocation_context=inv_ctx)  # type: ignore[arg-type]
    finally:
        runtime.logger.raw_logger.removeHandler(handler)

    # 1. Verify distributed parent -> child span hierarchy
    spans = exporter.get_finished_spans()
    span_by_name = {span.name: span for span in spans}
    assert "travel_concierge.invocation" in span_by_name
    assert "agent.travel_concierge_coordinator" in span_by_name
    assert "tool.search_budget_flights" in span_by_name

    root_span = span_by_name["travel_concierge.invocation"]
    agent_span = span_by_name["agent.travel_concierge_coordinator"]
    assert agent_span.parent is not None
    assert agent_span.parent.span_id == root_span.context.span_id
    assert agent_span.context.trace_id == root_span.context.trace_id

    # 2. Verify structured JSON Intent vs. Outcome logs with PII scrubbed
    parsed = [json.loads(line) for line in handler.lines]
    event_types = [entry["event_type"] for entry in parsed]
    assert "agent.intent" in event_types
    assert "agent.outcome" in event_types
    assert event_types.index("agent.intent") < event_types.index("agent.outcome")
    assert all("bob@example.com" not in line for line in handler.lines)
    assert any("[REDACTED_EMAIL]" in line for line in handler.lines)


def test_secret_provider_uses_secret_manager_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SecretProvider fetches from Secret Manager in prod and never falls back to literals."""

    class _FakeSecretClient:
        def access_secret_version(self, request: dict[str, str]) -> SimpleNamespace:
            assert request["name"] == (
                "projects/travel-prod-proj/secrets/travel-concierge-gemini-api-key/versions/latest"
            )
            return SimpleNamespace(payload=SimpleNamespace(data=b"resolved-from-sm"))

    base = AppSettings.from_env()
    sm_settings = AppSettings(
        **{
            **base.__dict__,
            "use_secret_manager": True,
            "gcp_project_id": "travel-prod-proj",
            "use_vertex_ai": False,
        }
    )
    provider = SecretProvider(settings=sm_settings, client=_FakeSecretClient())
    assert provider.resolve_gemini_api_key() == "resolved-from-sm"

    # Local mode fails closed if env var is unset (zero literal fallback secrets).
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    local_settings = AppSettings(
        **{**base.__dict__, "use_secret_manager": False, "use_vertex_ai": False}
    )
    local_provider = SecretProvider(settings=local_settings)
    with pytest.raises(SecretResolutionError):
        local_provider.resolve_gemini_api_key()


def test_fastapi_health_and_hitl_decision_endpoint(tmp_path: Path) -> None:
    """Verify FastAPI /health and /approvals/{id}/decision endpoints."""
    runtime = create_travel_concierge_runtime(settings=_isolated_settings(tmp_path))
    api = create_fastapi_app(runtime=runtime)
    client = TestClient(api)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    hold = runtime.toolset.request_booking_approval(
        traveler_id="traveler_api",
        flight_id="FLT-AUS-DEN-101",
        hotel_id="HTL-DEN-001",
        departure_date="2026-09-10",
        return_date="2026-09-13",
        travelers=1,
        total_cost_usd=626.25,
        trip_budget_usd=1500.0,
        summary="Denver API test hold",
    )
    approval_id = hold["approval_id"]

    decision = client.post(
        f"/approvals/{approval_id}/decision",
        json={"reviewer_id": "traveler_api", "approved": True},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"
