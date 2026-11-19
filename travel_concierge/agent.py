"""ADK multi-agent orchestration, strategic model routing, and App assembly.

Architecture:
    travel_concierge_coordinator (gemini-2.5-flash)
      ├── trip_planning_pipeline (SequentialAgent)
      │     ├── trip_intake_specialist (gemini-2.5-flash)
      │     └── itinerary_budget_architect (gemini-2.5-pro)
      └── booking_execution_specialist (gemini-2.5-flash + HITL confirmation)

Wrapped in an ADK ``App`` configured with:
- ``AgentOpsGuardrailPlugin`` (PII redaction, prompt-injection defense,
  budget guardrails, self-eval, Intent/Outcome JSON logging, OTel spans, async memory).
- ``EventsCompactionConfig`` (sliding-window history compaction).
- ``ContextCacheConfig`` (system instruction caching).
- ``ResumabilityConfig(is_resumable=True)`` (resumable HITL confirmations).
"""

from __future__ import annotations

from dataclasses import dataclass

from google.adk import Runner
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.base_session_service import BaseSessionService
from opentelemetry.sdk.trace.export import SpanExporter

from travel_concierge.config.settings import AppSettings, TaskComplexity
from travel_concierge.memory.async_memory import AsyncMemoryConsolidator
from travel_concierge.memory.compaction import (
    build_compaction_config,
    build_context_cache_config,
)
from travel_concierge.memory.store import (
    PersistentTravelStore,
    build_memory_service,
    build_session_service,
)
from travel_concierge.observability.logging import StructuredAgentLogger
from travel_concierge.observability.telemetry import TelemetryManager
from travel_concierge.prompts import (
    BOOKING_EXECUTION_INSTRUCTION,
    COORDINATOR_INSTRUCTION,
    GLOBAL_CONCIERGE_CONSTITUTION,
    ITINERARY_ARCHITECT_INSTRUCTION,
    TRIP_INTAKE_INSTRUCTION,
)
from travel_concierge.security.guardrails import AgentOpsGuardrailPlugin
from travel_concierge.security.pii_redactor import PIIRedactor
from travel_concierge.tools.travel_tools import TravelConciergeToolset


@dataclass
class TravelConciergeRuntime:
    """Container bundling the ADK App, Runner, services, store, and observability hooks."""

    settings: AppSettings
    store: PersistentTravelStore
    session_service: BaseSessionService
    memory_service: BaseMemoryService
    memory_consolidator: AsyncMemoryConsolidator
    redactor: PIIRedactor
    logger: StructuredAgentLogger
    telemetry: TelemetryManager
    guardrail_plugin: AgentOpsGuardrailPlugin
    toolset: TravelConciergeToolset
    root_agent: LlmAgent
    app: App
    runner: Runner


def create_travel_concierge_runtime(
    settings: AppSettings | None = None,
    *,
    span_exporter: SpanExporter | None = None,
    enable_context_cache: bool = True,
) -> TravelConciergeRuntime:
    """Build a complete, self-contained Travel Concierge ADK runtime.

    Args:
        settings: Optional ``AppSettings`` override (defaults to ``AppSettings.from_env()``).
        span_exporter: Optional OpenTelemetry ``SpanExporter`` (e.g., ``InMemorySpanExporter``
            in unit tests or evaluation runs).
        enable_context_cache: Enable Gemini context caching config (typically ``True`` on
            Vertex AI / production workloads).
    """
    cfg = settings or AppSettings.from_env()
    redactor = PIIRedactor(settings=cfg)
    logger = StructuredAgentLogger(level=cfg.log_level, redactor=redactor)
    telemetry = TelemetryManager(settings=cfg, exporter=span_exporter, redactor=redactor)

    store = PersistentTravelStore(cfg.memory_db_path, redactor=redactor)
    session_service = build_session_service(cfg)
    memory_service = build_memory_service(cfg)
    memory_consolidator = AsyncMemoryConsolidator(
        store=store,
        memory_service=memory_service,
        redactor=redactor,
        logger=logger,
    )

    toolset = TravelConciergeToolset(
        store=store,
        memory_consolidator=memory_consolidator,
        settings=cfg,
    )

    # Strategic Model Routing: Flash for fast intake/routing/execution, Pro for deep planning.
    fast_model = cfg.model_routing.select_model(TaskComplexity.FAST_ROUTING)
    planning_model = cfg.model_routing.select_model(TaskComplexity.COMPLEX_PLANNING)

    trip_intake_specialist = LlmAgent(
        name="trip_intake_specialist",
        model=fast_model,
        description=(
            "Recalls and saves traveler preferences and searches verified flights and hotels."
        ),
        instruction=TRIP_INTAKE_INSTRUCTION,
        tools=[
            toolset.recall_traveler_preferences,
            toolset.save_traveler_preference,
            toolset.search_budget_flights,
            toolset.search_hotels_within_budget,
        ],
        output_key="intake_quotes",
    )

    itinerary_budget_architect = LlmAgent(
        name="itinerary_budget_architect",
        model=planning_model,
        description=(
            "Synthesizes multi-day itineraries and performs deterministic trip budget optimization."
        ),
        instruction=ITINERARY_ARCHITECT_INSTRUCTION,
        tools=[
            toolset.calculate_trip_budget_breakdown,
            toolset.search_budget_flights,
            toolset.search_hotels_within_budget,
        ],
        output_key="recommended_itinerary",
    )

    trip_planning_pipeline = SequentialAgent(
        name="trip_planning_pipeline",
        description=(
            "Sequential planning pipeline: recalls traveler preferences and gathers quotes "
            "(Fast model), then optimizes itinerary and verifies budget compliance (Pro model)."
        ),
        sub_agents=[trip_intake_specialist, itinerary_budget_architect],
    )

    booking_execution_specialist = LlmAgent(
        name="booking_execution_specialist",
        model=fast_model,
        description=(
            "Creates pending Human-in-the-Loop booking holds and executes verified reservations "
            "only after explicit human confirmation."
        ),
        instruction=BOOKING_EXECUTION_INSTRUCTION,
        tools=[
            toolset.request_booking_approval,
            toolset.build_booking_function_tool(),
        ],
        output_key="booking_status",
    )

    coordinator = LlmAgent(
        name="travel_concierge_coordinator",
        model=fast_model,
        description="Root coordinator routing travel planning, preference memory, and HITL booking.",
        global_instruction=GLOBAL_CONCIERGE_CONSTITUTION,
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            toolset.recall_traveler_preferences,
            toolset.save_traveler_preference,
        ],
        sub_agents=[trip_planning_pipeline, booking_execution_specialist],
    )

    guardrail_plugin = AgentOpsGuardrailPlugin(
        settings=cfg,
        redactor=redactor,
        logger=logger,
        telemetry=telemetry,
        memory_consolidator=memory_consolidator,
    )

    adk_app = App(
        name=cfg.app_name,
        root_agent=coordinator,
        plugins=[guardrail_plugin],
        events_compaction_config=build_compaction_config(cfg, redactor=redactor),
        context_cache_config=build_context_cache_config() if enable_context_cache else None,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )

    runner = Runner(
        app=adk_app,
        session_service=session_service,
        memory_service=memory_service,
        auto_create_session=True,
    )

    return TravelConciergeRuntime(
        settings=cfg,
        store=store,
        session_service=session_service,
        memory_service=memory_service,
        memory_consolidator=memory_consolidator,
        redactor=redactor,
        logger=logger,
        telemetry=telemetry,
        guardrail_plugin=guardrail_plugin,
        toolset=toolset,
        root_agent=coordinator,
        app=adk_app,
        runner=runner,
    )


# Default module-level symbols discovered automatically by `adk web`, `adk run`, and `adk eval`.
_default_runtime = create_travel_concierge_runtime()
root_agent = _default_runtime.root_agent
app = _default_runtime.app
