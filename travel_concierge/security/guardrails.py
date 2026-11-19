"""ADK Guardrails, Self-Evaluation, PII Redaction, Intent/Outcome Logging, and Tracing Plugin.

Implements a production ``BasePlugin`` registered on the ADK ``App`` so every
agent invocation, LLM request/response, and tool execution passes through:
1. Pre-LLM PII scrubbing & prompt-injection containment.
2. Distributed OpenTelemetry parent-child span management.
3. Explicit Intent logging BEFORE execution and Outcome logging AFTER execution.
4. Deterministic policy guardrails (budget ceilings, fail-closed HITL enforcement).
5. Post-model self-evaluation verifying response safety and factual grounding.
6. Non-blocking async background memory consolidation on ``after_run_callback``.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import trace
from opentelemetry.trace import Span

from travel_concierge.config.settings import AppSettings
from travel_concierge.observability.logging import StructuredAgentLogger
from travel_concierge.observability.telemetry import TelemetryManager
from travel_concierge.schemas import StructuredToolError
from travel_concierge.security.pii_redactor import PIIRedactor

if TYPE_CHECKING:
    from travel_concierge.memory.async_memory import AsyncMemoryConsolidator

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"bypass\s+(?:all\s+)?(?:rules|guardrails|policy|approval|hitl|confirmation)", re.I),
    re.compile(r"auto[- ]?approve\s+(?:this|the)\s+(?:booking|reservation|payment)", re.I),
    re.compile(r"(?:reveal|print|dump|show)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions)", re.I),
    re.compile(r"override\s+(?:the\s+)?(?:budget|ceiling|limit|security)", re.I),
    re.compile(r"pretend\s+the\s+(?:human|user)\s+(?:already\s+)?approved", re.I),
)

_UNVERIFIED_BOOKING_CLAIM_RE = re.compile(
    r"\b(?:booking\s+(?:is\s+)?confirmed|reservation\s+confirmed|charged\s+your\s+card|"
    r"confirmation\s+code\s*[:#]?\s*TRP-[A-Z0-9]+)\b",
    re.I,
)


def detect_prompt_injection(text: str) -> str | None:
    """Return the matched rule snippet if text resembles a prompt-injection attempt."""
    for pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


class AgentOpsGuardrailPlugin(BasePlugin):
    """Unified AgentOps ADK Plugin enforcing security, self-eval, logging, and OTel tracing."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        redactor: PIIRedactor | None = None,
        logger: StructuredAgentLogger | None = None,
        telemetry: TelemetryManager | None = None,
        memory_consolidator: AsyncMemoryConsolidator | None = None,
    ) -> None:
        super().__init__(name="agentops_guardrails_plugin")
        self._settings = settings or AppSettings.from_env()
        self._redactor = redactor or PIIRedactor(settings=self._settings)
        self._logger = logger or StructuredAgentLogger(
            level=self._settings.log_level, redactor=self._redactor
        )
        self._telemetry = telemetry or TelemetryManager(
            settings=self._settings, redactor=self._redactor
        )
        self._memory_consolidator = memory_consolidator
        self._active_spans: dict[str, tuple[Span, float]] = {}

    def _span_key(self, scope: str, identifier: str) -> str:
        return f"{scope}:{identifier}"

    def _start_tracked_span(
        self,
        key: str,
        span_name: str,
        *,
        attributes: dict[str, Any] | None = None,
        parent_key: str | None = None,
    ) -> Span:
        parent_ctx = None
        if parent_key and parent_key in self._active_spans:
            parent_span = self._active_spans[parent_key][0]
            parent_ctx = trace.set_span_in_context(parent_span)
        span = self._telemetry.start_span(
            span_name,
            attributes=attributes,
            parent_context=parent_ctx,
        )
        self._active_spans[key] = (span, time.perf_counter())
        return span

    def _end_tracked_span(
        self,
        key: str,
        *,
        action: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> float | None:
        entry = self._active_spans.pop(key, None)
        if entry is None:
            return None
        span, started_at = entry
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        self._telemetry.record_outcome_event(
            span,
            action=action,
            status=status,
            details={"latency_ms": round(latency_ms, 2), **(details or {})},
        )
        span.end()
        return latency_ms

    # ------------------------------------------------------------------
    # 1. User Ingress: PII Scrubbing + Prompt-Injection Detection
    # ------------------------------------------------------------------

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Scrub PII and flag prompt-injection attempts before session storage or LLM calls."""
        if not user_message or not user_message.parts:
            return None

        modified = False
        new_parts: list[types.Part] = []
        injection_detected: str | None = None
        redacted_categories: set[str] = set()

        for part in user_message.parts:
            if part.text:
                inj = detect_prompt_injection(part.text)
                if inj:
                    injection_detected = inj

                res = self._redactor.redact_text(part.text)
                if res.had_pii:
                    modified = True
                    redacted_categories.update(res.redacted_categories)
                new_parts.append(types.Part(text=res.redacted_text))
            else:
                new_parts.append(part)

        state = invocation_context.session.state
        if redacted_categories:
            state["security:last_pii_redacted"] = sorted(redacted_categories)
            self._logger.log_security_event(
                rule="pii_ingress_redaction",
                action_taken="redacted_before_llm",
                severity="INFO",
                details={
                    "session_id": invocation_context.session.id,
                    "categories": sorted(redacted_categories),
                },
            )

        if injection_detected:
            state["security:prompt_injection_detected"] = True
            state["security:prompt_injection_pattern"] = injection_detected
            self._logger.log_security_event(
                rule="prompt_injection_defense",
                action_taken="blocked_before_llm",
                severity="WARNING",
                details={
                    "session_id": invocation_context.session.id,
                    "matched_pattern": injection_detected,
                },
            )
        else:
            state["security:prompt_injection_detected"] = False

        if modified:
            return types.Content(role=user_message.role or "user", parts=new_parts)
        return None

    # ------------------------------------------------------------------
    # 2. Invocation Lifecycle: Root Trace + Async Memory Consolidation
    # ------------------------------------------------------------------

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        """Start the root OpenTelemetry span for this user turn."""
        inv_id = invocation_context.invocation_id
        span = self._start_tracked_span(
            self._span_key("run", inv_id),
            "travel_concierge.invocation",
            attributes={
                "session.id": invocation_context.session.id,
                "user.id": invocation_context.user_id,
                "invocation.id": inv_id,
            },
        )
        self._telemetry.record_intent_event(
            span,
            action="process_user_request",
            details={"session_id": invocation_context.session.id},
        )
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Complete root OTel span and trigger non-blocking async memory consolidation."""
        inv_id = invocation_context.invocation_id
        self._end_tracked_span(
            self._span_key("run", inv_id),
            action="process_user_request",
            status="ok",
            details={"session_id": invocation_context.session.id},
        )
        if self._memory_consolidator is not None:
            self._memory_consolidator.schedule_session_consolidation(
                session=invocation_context.session,
                traveler_id=invocation_context.user_id,
            )

    # ------------------------------------------------------------------
    # 3. Sub-Agent Delegation Spans
    # ------------------------------------------------------------------

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        inv_id = callback_context.invocation_id
        model_name = getattr(agent, "model", "workflow_orchestrator")
        self._start_tracked_span(
            self._span_key("agent", f"{inv_id}:{agent.name}"),
            f"agent.{agent.name}",
            parent_key=self._span_key("run", inv_id),
            attributes={
                "agent.name": agent.name,
                "agent.model": str(model_name),
                "session.id": callback_context.session.id,
            },
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        inv_id = callback_context.invocation_id
        self._end_tracked_span(
            self._span_key("agent", f"{inv_id}:{agent.name}"),
            action=f"agent.{agent.name}",
            status="ok",
        )
        return None

    # ------------------------------------------------------------------
    # 4. Pre-Model Defense & Post-Model Self-Evaluation Guardrail
    # ------------------------------------------------------------------

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """Short-circuit prompt injections, scrub outgoing context, and log LLM intent."""
        state = callback_context.state
        if state.get("security:prompt_injection_detected"):
            state["security:prompt_injection_detected"] = False
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "Security Guardrail Activated: I cannot ignore safety policies, "
                                "override your trip budget ceiling, or auto-approve bookings "
                                "without explicit human confirmation. Please restate your travel "
                                "planning request and I will be glad to help."
                            )
                        )
                    ],
                )
            )

        # Defense-in-depth PII redaction across all request contents before Gemini sees them.
        if llm_request.contents:
            for content in llm_request.contents:
                if not content.parts:
                    continue
                for part in content.parts:
                    if part.text:
                        scrubbed = self._redactor.redact_text(part.text)
                        if scrubbed.had_pii:
                            part.text = scrubbed.redacted_text

        inv_id = callback_context.invocation_id
        agent_name = callback_context.agent_name
        model_name = llm_request.model or "default"

        self._logger.log_intent(
            actor=agent_name,
            action=f"llm.generate:{model_name}",
            session_id=callback_context.session.id,
            invocation_id=inv_id,
            intended_args={"model": model_name, "content_turns": len(llm_request.contents or [])},
            rationale="model_reasoning_step",
        )
        span = self._start_tracked_span(
            self._span_key("model", f"{inv_id}:{agent_name}"),
            f"llm.{agent_name}",
            parent_key=self._span_key("agent", f"{inv_id}:{agent_name}"),
            attributes={"llm.model": model_name, "agent.name": agent_name},
        )
        self._telemetry.record_intent_event(
            span, action="llm.generate", details={"model": model_name}
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Perform post-model self-evaluation (PII leak check + unapproved booking claim check)."""
        inv_id = callback_context.invocation_id
        agent_name = callback_context.agent_name
        state = callback_context.state

        self_eval_passed = True
        self_eval_notes: list[str] = []
        mutated = False

        if llm_response.content and llm_response.content.parts:
            confirmed_code = state.get("booking:last_confirmed_code")
            for part in llm_response.content.parts:
                if not part.text:
                    continue
                redaction = self._redactor.redact_text(part.text)
                if redaction.had_pii:
                    part.text = redaction.redacted_text
                    mutated = True
                    self_eval_passed = False
                    self_eval_notes.append(
                        f"scrubbed_output_pii:{','.join(redaction.redacted_categories)}"
                    )

                if _UNVERIFIED_BOOKING_CLAIM_RE.search(part.text) and not confirmed_code:
                    self_eval_passed = False
                    self_eval_notes.append("blocked_unverified_booking_confirmation_claim")
                    part.text = (
                        "Policy Self-Evaluation Notice: No booking has been executed yet because "
                        "explicit human confirmation has not been recorded. Please review the "
                        "pending approval details and confirm before I finalize any reservation."
                    )
                    mutated = True

        latency_ms = self._end_tracked_span(
            self._span_key("model", f"{inv_id}:{agent_name}"),
            action="llm.generate",
            status="ok" if self_eval_passed else "corrected",
            details={"self_eval_passed": self_eval_passed, "notes": self_eval_notes},
        )
        self._logger.log_outcome(
            actor=agent_name,
            action="llm.generate",
            outcome_status="ok" if self_eval_passed else "self_eval_corrected",
            latency_ms=latency_ms,
            session_id=callback_context.session.id,
            invocation_id=inv_id,
            outcome_summary={
                "self_eval_passed": self_eval_passed,
                "self_eval_notes": self_eval_notes,
            },
        )
        return llm_response if mutated else None

    # ------------------------------------------------------------------
    # 5. Tool Policy Guardrails, Intent vs. Outcome Capture, Guided Errors
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Record tool Intent, enforce policy ceilings, and start linked tool OTel span."""
        inv_id = tool_context.invocation_id
        agent_name = tool_context.agent_name
        call_key = tool_context.function_call_id or tool.name

        self._logger.log_intent(
            actor=agent_name,
            action=f"tool.{tool.name}",
            session_id=tool_context.session.id,
            invocation_id=inv_id,
            intended_args=tool_args,
            rationale="tool_invocation",
        )
        span = self._start_tracked_span(
            self._span_key("tool", f"{inv_id}:{call_key}"),
            f"tool.{tool.name}",
            parent_key=self._span_key("agent", f"{inv_id}:{agent_name}"),
            attributes={
                "tool.name": tool.name,
                "agent.name": agent_name,
            },
        )
        self._telemetry.record_intent_event(
            span, action=f"tool.{tool.name}", details={"args": tool_args}
        )

        # Deterministic policy check: reject budgets exceeding the hard enterprise safety cap.
        for budget_key in ("total_budget_usd", "total_cost_usd", "max_price_per_person_usd"):
            raw_val = tool_args.get(budget_key)
            if isinstance(raw_val, (int, float)) and raw_val > self._settings.max_single_trip_budget_usd:
                self._logger.log_security_event(
                    rule="max_trip_budget_ceiling",
                    action_taken="blocked_tool_call",
                    details={
                        "tool": tool.name,
                        "field": budget_key,
                        "limit_usd": self._settings.max_single_trip_budget_usd,
                    },
                )
                self._end_tracked_span(
                    self._span_key("tool", f"{inv_id}:{call_key}"),
                    action=f"tool.{tool.name}",
                    status="blocked",
                    details={"error_code": "POLICY_BUDGET_CAP_EXCEEDED"},
                )
                return StructuredToolError(
                    error_code="POLICY_BUDGET_CAP_EXCEEDED",
                    message=(
                        f"Requested amount (${raw_val:,.2f}) exceeds the maximum allowed "
                        f"single-trip policy cap of ${self._settings.max_single_trip_budget_usd:,.2f}."
                    ),
                    recovery_instructions=(
                        "Inform the traveler of the policy ceiling and ask them to reduce the "
                        "budget or split the itinerary into smaller trips."
                    ),
                    retryable=True,
                ).model_dump()

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record tool Outcome, scrub any PII from tool output, and close tool OTel span."""
        inv_id = tool_context.invocation_id
        agent_name = tool_context.agent_name
        call_key = tool_context.function_call_id or tool.name

        scrubbed_result, pii_found = self._redactor.redact_mapping(result)
        outcome_status = (
            str(scrubbed_result.get("status", "ok"))
            if isinstance(scrubbed_result, dict)
            else "ok"
        )
        error_code = (
            scrubbed_result.get("error_code") if isinstance(scrubbed_result, dict) else None
        )

        latency_ms = self._end_tracked_span(
            self._span_key("tool", f"{inv_id}:{call_key}"),
            action=f"tool.{tool.name}",
            status=outcome_status,
            details={"error_code": error_code, "pii_redacted": bool(pii_found)},
        )
        self._logger.log_outcome(
            actor=agent_name,
            action=f"tool.{tool.name}",
            outcome_status=outcome_status,
            latency_ms=latency_ms,
            session_id=tool_context.session.id,
            invocation_id=inv_id,
            error_code=error_code,
            outcome_summary={
                "status": outcome_status,
                "result_keys": sorted(scrubbed_result.keys())
                if isinstance(scrubbed_result, dict)
                else [],
                "pii_redacted": list(pii_found),
            },
        )
        return scrubbed_result if pii_found and isinstance(scrubbed_result, dict) else None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Convert unexpected tool exceptions into structured LLM recovery instructions."""
        inv_id = tool_context.invocation_id
        agent_name = tool_context.agent_name
        call_key = tool_context.function_call_id or tool.name

        latency_ms = self._end_tracked_span(
            self._span_key("tool", f"{inv_id}:{call_key}"),
            action=f"tool.{tool.name}",
            status="error",
            details={"error_type": type(error).__name__},
        )
        self._logger.log_outcome(
            actor=agent_name,
            action=f"tool.{tool.name}",
            outcome_status="error",
            latency_ms=latency_ms,
            session_id=tool_context.session.id,
            invocation_id=inv_id,
            error_code="UNHANDLED_TOOL_EXCEPTION",
            outcome_summary={"error_type": type(error).__name__},
        )
        return StructuredToolError(
            error_code="TOOL_EXECUTION_FAILED",
            message=f"Tool '{tool.name}' encountered an unexpected internal error.",
            recovery_instructions=(
                "Verify all parameter formats match the tool schema and retry once. "
                "If the error persists, inform the traveler transparently without fabricating data."
            ),
            retryable=True,
            details={"error_type": type(error).__name__},
        ).model_dump()

    async def close(self) -> None:
        """Flush telemetry spans and drain async memory tasks cleanly."""
        if self._memory_consolidator is not None:
            await self._memory_consolidator.drain()
        self._telemetry.shutdown()
