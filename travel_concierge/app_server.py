"""FastAPI HTTP service exposing health checks, chat sessions, and HITL approval decisions.

Security defaults:
- Local development binds exclusively to ``127.0.0.1`` (Cloud Run overrides via ``PORT``).
- Strict Pydantic request validation on all endpoints.
- Generic error responses (never leaks stack traces or internal DB details).
- PII scrubbing applied before logging or returning messages.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from travel_concierge.agent import TravelConciergeRuntime, create_travel_concierge_runtime
from travel_concierge.config.settings import AppSettings


class ChatTurnRequest(BaseModel):
    """Validated chat turn payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    traveler_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    session_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    message: str = Field(min_length=1, max_length=4000)


class ChatTurnResponse(BaseModel):
    """Structured chat response returned to the client."""

    session_id: str
    traveler_id: str
    reply: str


class HumanApprovalDecisionRequest(BaseModel):
    """Explicit human approval or rejection for a pending HITL booking hold."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reviewer_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    approved: bool


def create_fastapi_app(
    runtime: TravelConciergeRuntime | None = None,
) -> FastAPI:
    """Create the FastAPI application wired to the Travel Concierge ADK runtime."""
    active_runtime = runtime or create_travel_concierge_runtime()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await active_runtime.guardrail_plugin.close()

    api = FastAPI(
        title="Smart Travel & Trip Budget Concierge Agent",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @api.get("/health")
    async def health() -> dict[str, str]:
        """Liveness & readiness check for Cloud Run / Agent Runtime."""
        return {
            "status": "healthy",
            "app": active_runtime.settings.app_name,
            "fast_model": active_runtime.settings.model_routing.fast_model,
            "planning_model": active_runtime.settings.model_routing.planning_model,
        }

    @api.post("/approvals/{approval_id}/decision")
    async def submit_human_approval(
        approval_id: str,
        body: HumanApprovalDecisionRequest,
    ) -> dict[str, str]:
        """Record an explicit human Approve/Reject decision for a pending booking hold."""
        try:
            updated = active_runtime.store.record_human_decision(
                approval_id=approval_id,
                approved=body.approved,
                reviewer_id=body.reviewer_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Approval hold not found.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Approval hold is expired or already finalized.",
            ) from exc

        active_runtime.logger.log_security_event(
            rule="hitl_human_decision",
            action_taken=str(updated["status"]),
            severity="INFO",
            details={
                "approval_id": approval_id,
                "reviewer_id": body.reviewer_id,
            },
        )
        return {
            "approval_id": str(updated["approval_id"]),
            "status": str(updated["status"]),
            "approved_by": str(updated["approved_by"]),
        }

    @api.post("/chat", response_model=ChatTurnResponse)
    async def chat_turn(body: ChatTurnRequest) -> ChatTurnResponse:
        """Execute a single conversational turn via the ADK Runner."""
        user_msg = types.Content(role="user", parts=[types.Part(text=body.message)])
        collected_text: list[str] = []
        try:
            async for event in active_runtime.runner.run_async(
                user_id=body.traveler_id,
                session_id=body.session_id,
                new_message=user_msg,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text and event.author != "user":
                            collected_text.append(part.text)
        except Exception as exc:
            active_runtime.logger.emit(
                "api.chat_error",
                f"Chat execution failed: {type(exc).__name__}",
                payload={"session_id": body.session_id},
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Unable to process travel request at this time.",
            ) from exc

        reply = "\n".join(collected_text).strip() or "No response generated."
        return ChatTurnResponse(
            session_id=body.session_id,
            traveler_id=body.traveler_id,
            reply=active_runtime.redactor.redact_text(reply).redacted_text,
        )

    return api


def main() -> None:
    """Run local HTTP service bound strictly to loopback unless APP_HOST is configured."""
    import uvicorn

    settings = AppSettings.from_env()
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(create_fastapi_app(create_travel_concierge_runtime(settings)), host=host, port=port)


if __name__ == "__main__":
    main()
