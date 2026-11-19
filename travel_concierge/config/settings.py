"""Runtime configuration and strategic model routing for the Travel Concierge Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class TaskComplexity(StrEnum):
    """Task complexity tiers used for strategic model routing."""

    FAST_ROUTING = "fast_routing"
    TOOL_EXECUTION = "tool_execution"
    COMPLEX_PLANNING = "complex_planning"
    COMPACTION_SUMMARY = "compaction_summary"


@dataclass(frozen=True)
class ModelRoutingConfig:
    """Maps agent roles and task tiers to the optimal Gemini model.

    Uses fast, low-latency Flash models for routing, intake, tool orchestration,
    and booking confirmation, while reserving higher-reasoning Pro models for
    multi-constraint itinerary synthesis and budget trade-off optimization.
    """

    fast_model: str = "gemini-2.5-flash"
    planning_model: str = "gemini-2.5-pro"

    def select_model(self, complexity: TaskComplexity) -> str:
        """Select the appropriate Gemini model identifier for a given task tier."""
        if complexity == TaskComplexity.COMPLEX_PLANNING:
            return self.planning_model
        return self.fast_model


@dataclass(frozen=True)
class AppSettings:
    """Validated runtime settings loaded exclusively from environment variables."""

    app_name: str
    environment: str
    gcp_project_id: str | None
    gcp_location: str
    use_vertex_ai: bool
    use_secret_manager: bool
    gemini_api_key_secret_name: str
    storage_backend: str
    session_db_path: Path
    memory_db_path: Path
    vertex_agent_engine_id: str | None
    enable_cloud_dlp: bool
    max_single_trip_budget_usd: float
    approval_ttl_seconds: int
    compaction_interval: int
    compaction_overlap_size: int
    compaction_token_threshold: int
    compaction_event_retention_size: int
    log_level: str
    otel_service_name: str
    otel_export_cloud_trace: bool
    model_routing: ModelRoutingConfig

    @classmethod
    def from_env(cls) -> AppSettings:
        """Load application settings from environment variables with safe defaults."""
        fast_model = os.getenv("FAST_MODEL_NAME", "gemini-2.5-flash").strip()
        planning_model = os.getenv("PLANNING_MODEL_NAME", "gemini-2.5-pro").strip()
        return cls(
            app_name=os.getenv("ADK_APP_NAME", "travel_concierge").strip(),
            environment=os.getenv("APP_ENV", "local").strip().lower(),
            gcp_project_id=os.getenv("GOOGLE_CLOUD_PROJECT") or None,
            gcp_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip(),
            use_vertex_ai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true",
            use_secret_manager=os.getenv("USE_SECRET_MANAGER", "false").lower() == "true",
            gemini_api_key_secret_name=os.getenv(
                "GEMINI_API_KEY_SECRET_NAME", "travel-concierge-gemini-api-key"
            ).strip(),
            storage_backend=os.getenv("AGENT_STORAGE_BACKEND", "sqlite").strip().lower(),
            session_db_path=Path(os.getenv("SESSION_DB_PATH", "./data/travel_sessions.db")),
            memory_db_path=Path(os.getenv("MEMORY_DB_PATH", "./data/travel_memory.db")),
            vertex_agent_engine_id=os.getenv("VERTEX_AGENT_ENGINE_ID") or None,
            enable_cloud_dlp=os.getenv("ENABLE_CLOUD_DLP", "false").lower() == "true",
            max_single_trip_budget_usd=float(
                os.getenv("MAX_SINGLE_TRIP_BUDGET_USD", "25000.0")
            ),
            approval_ttl_seconds=int(os.getenv("APPROVAL_TTL_SECONDS", "900")),
            compaction_interval=int(os.getenv("COMPACTION_INTERVAL", "6")),
            compaction_overlap_size=int(os.getenv("COMPACTION_OVERLAP_SIZE", "2")),
            compaction_token_threshold=int(os.getenv("COMPACTION_TOKEN_THRESHOLD", "4000")),
            compaction_event_retention_size=int(
                os.getenv("COMPACTION_EVENT_RETENTION_SIZE", "4")
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            otel_service_name=os.getenv(
                "OTEL_SERVICE_NAME", "smart-travel-concierge-agent"
            ).strip(),
            otel_export_cloud_trace=(
                os.getenv("OTEL_EXPORT_CLOUD_TRACE", "false").lower() == "true"
            ),
            model_routing=ModelRoutingConfig(
                fast_model=fast_model,
                planning_model=planning_model,
            ),
        )
