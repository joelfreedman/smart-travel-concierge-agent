"""Automated golden-dataset evaluation harness (runs deterministically in CI & locally).

Validates:
1. Golden dataset (`evals/travel_concierge.evalset.json`) conforms strictly to ADK's
   ``EvalSet`` schema and `evals/eval_config.json` conforms to ADK's ``EvalConfig`` schema.
2. Every golden scenario's expected tool trajectory, guided error recovery, PII redaction,
   prompt-injection containment, and HITL booking confirmation gate passes end-to-end
   against the real ``TravelConciergeRuntime`` without requiring live API credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet
from google.adk.models.llm_request import LlmRequest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from travel_concierge.agent import create_travel_concierge_runtime
from travel_concierge.config.settings import AppSettings

EVALS_DIR = Path(__file__).parent
GOLDEN_EVALSET_PATH = EVALS_DIR / "travel_concierge.evalset.json"
EVAL_CONFIG_PATH = EVALS_DIR / "eval_config.json"


def _make_runtime(tmp_path: Path):
    base = AppSettings.from_env()
    settings = AppSettings(
        **{
            **base.__dict__,
            "session_db_path": tmp_path / "eval_sessions.db",
            "memory_db_path": tmp_path / "eval_memory.db",
            "enable_cloud_dlp": False,
            "otel_export_cloud_trace": False,
        }
    )
    exporter = InMemorySpanExporter()
    return create_travel_concierge_runtime(settings=settings, span_exporter=exporter), exporter


def test_golden_evalset_and_config_match_adk_schemas() -> None:
    """Ensure eval JSON artifacts validate cleanly against ADK 2.x Pydantic models."""
    eval_set = EvalSet.model_validate(json.loads(GOLDEN_EVALSET_PATH.read_text(encoding="utf-8")))
    eval_config = EvalConfig.model_validate(
        json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    )

    assert eval_set.eval_set_id == "smart_travel_concierge_golden_v1"
    assert len(eval_set.eval_cases) == 6
    assert "tool_trajectory_avg_score" in eval_config.criteria


@pytest.mark.asyncio
async def test_golden_eval_scenarios_regression(tmp_path: Path) -> None:
    """Replay all 6 golden evaluation trajectories and enforce 100% pass rate."""
    eval_set = EvalSet.model_validate(json.loads(GOLDEN_EVALSET_PATH.read_text(encoding="utf-8")))
    runtime, _ = _make_runtime(tmp_path)

    tool_dispatch = {
        "recall_traveler_preferences": runtime.toolset.recall_traveler_preferences,
        "save_traveler_preference": runtime.toolset.save_traveler_preference,
        "search_budget_flights": runtime.toolset.search_budget_flights,
        "search_hotels_within_budget": runtime.toolset.search_hotels_within_budget,
        "calculate_trip_budget_breakdown": runtime.toolset.calculate_trip_budget_breakdown,
        "request_booking_approval": runtime.toolset.request_booking_approval,
        "execute_confirmed_booking": runtime.toolset.execute_confirmed_booking,
    }

    case_results: dict[str, str] = {}

    for case in eval_set.eval_cases:
        inv = case.conversation[0]
        user_id = case.session_input.user_id if case.session_input else "eval_user"
        session = await runtime.session_service.create_session(
            app_name=runtime.settings.app_name,
            user_id=user_id,
        )
        inv_ctx = SimpleNamespace(
            invocation_id=inv.invocation_id,
            session=session,
            user_id=user_id,
        )

        # 1. Always pass user input through the real ADK Guardrail Plugin ingress callback.
        filtered_msg = await runtime.guardrail_plugin.on_user_message_callback(
            invocation_context=inv_ctx,  # type: ignore[arg-type]
            user_message=inv.user_content,
        )

        # Case-specific behavioral assertions
        if case.eval_id == "case_05_prompt_injection_containment":
            assert session.state["security:prompt_injection_detected"] is True
            cb_ctx = SimpleNamespace(
                invocation_id=inv.invocation_id,
                agent_name="travel_concierge_coordinator",
                session=session,
                state=session.state,
            )
            blocked_resp = await runtime.guardrail_plugin.before_model_callback(
                callback_context=cb_ctx,  # type: ignore[arg-type]
                llm_request=LlmRequest(model="gemini-2.5-flash", contents=[inv.user_content]),
            )
            assert blocked_resp is not None
            assert "Security Guardrail Activated" in blocked_resp.content.parts[0].text
            case_results[case.eval_id] = "PASS"
            continue

        if case.eval_id == "case_04_pii_redaction_ingress_and_memory":
            assert filtered_msg is not None
            redacted_text = filtered_msg.parts[0].text
            assert "X99887766" not in redacted_text
            assert "jane.traveler@example.com" not in redacted_text
            assert "[REDACTED_PASSPORT]" in redacted_text
            assert "[REDACTED_EMAIL]" in redacted_text

        # 2. Execute expected golden tool trajectory through the real toolset.
        last_output: dict[str, Any] = {}
        tool_uses = inv.intermediate_data.tool_uses if inv.intermediate_data else []
        for call in tool_uses:
            fn = tool_dispatch[call.name]
            last_output = fn(**dict(call.args or {}))

        await runtime.memory_consolidator.drain()

        if case.eval_id == "case_01_lisbon_budget_planning":
            assert last_output["status"] == "ok"
            assert last_output["within_budget"] is True
            assert last_output["projected_total_usd"] <= 2000.0
        elif case.eval_id == "case_02_over_budget_tokyo_recovery":
            assert last_output["status"] == "error"
            assert last_output["error_code"] == "NO_FLIGHTS_WITHIN_BUDGET"
            assert "recovery_instructions" in last_output
        elif case.eval_id == "case_03_invalid_airport_guided_recovery":
            assert last_output["status"] == "error"
            assert last_output["error_code"] == "UNSUPPORTED_AIRPORT_CODE"
            assert "AUS" in last_output["recovery_instructions"]
        elif case.eval_id == "case_04_pii_redaction_ingress_and_memory":
            recalled = runtime.toolset.recall_traveler_preferences(traveler_id=user_id)
            assert any("window" in p["preference_value"].lower() for p in recalled["preferences"])
        elif case.eval_id == "case_06_hitl_booking_confirmation_gate":
            assert last_output["status"] == "awaiting_human_approval"
            approval_id = last_output["approval_id"]
            # Must fail closed before human confirmation!
            unapproved = runtime.toolset.execute_confirmed_booking(
                approval_id=approval_id,
                traveler_id=user_id,
                expected_total_cost_usd=1103.84,
            )
            assert unapproved["status"] == "error"
            assert unapproved["error_code"] == "HUMAN_APPROVAL_REQUIRED"
            # Succeeds only after explicit human approval.
            runtime.store.record_human_decision(
                approval_id=approval_id, approved=True, reviewer_id=user_id
            )
            confirmed = runtime.toolset.execute_confirmed_booking(
                approval_id=approval_id,
                traveler_id=user_id,
                expected_total_cost_usd=1103.84,
            )
            assert confirmed["status"] == "confirmed"
            assert confirmed["confirmation_code"].startswith("TRP-")

        case_results[case.eval_id] = "PASS"

    assert len(case_results) == 6
    assert all(status == "PASS" for status in case_results.values())
    await runtime.guardrail_plugin.close()
