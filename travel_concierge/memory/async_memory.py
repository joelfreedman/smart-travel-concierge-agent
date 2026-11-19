"""Asynchronous background memory extraction and consolidation.

Prevents conversation turns and UI responses from blocking on preference
extraction, PII scrubbing, SQLite persistence, or Vertex AI Memory Bank
ingestion by scheduling consolidation work via ``asyncio.create_task()``.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.session import Session

from travel_concierge.memory.store import PersistentTravelStore
from travel_concierge.observability.logging import StructuredAgentLogger
from travel_concierge.security.pii_redactor import PIIRedactor

# Lightweight deterministic patterns for zero-latency background preference mining.
_PREFERENCE_EXTRACTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "seating",
        re.compile(r"\b(?:prefer|like|want|always book)\s+(window|aisle|extra legroom)\s+seats?\b", re.I),
    ),
    (
        "airline",
        re.compile(r"\b(?:prefer|only fly|want)\s+(nonstop|direct)\s+flights?\b", re.I),
    ),
    (
        "dietary",
        re.compile(r"\b(?:i am|i'm|prefer|need)\s+(vegetarian|vegan|gluten[- ]free|halal|kosher)\b", re.I),
    ),
    (
        "lodging",
        re.compile(
            r"\b(?:prefer|stay near|want a hotel near|love)\s+"
            r"(transit|subway|train station|quiet neighborhoods?|free breakfast|boutique hotels?)\b",
            re.I,
        ),
    ),
    (
        "budget_style",
        re.compile(r"\b(budget[- ]conscious|backpacker|mid[- ]range|luxury|free cancellation)\b", re.I),
    ),
)


class AsyncMemoryConsolidator:
    """Manages non-blocking background memory extraction and session consolidation."""

    def __init__(
        self,
        store: PersistentTravelStore,
        *,
        memory_service: BaseMemoryService | None = None,
        redactor: PIIRedactor | None = None,
        logger: StructuredAgentLogger | None = None,
        max_concurrent_tasks: int = 8,
    ) -> None:
        self._store = store
        self._memory_service = memory_service
        self._redactor = redactor or PIIRedactor()
        self._logger = logger or StructuredAgentLogger()
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def pending_task_count(self) -> int:
        """Number of background memory tasks currently in flight."""
        return len(self._tasks)

    def _track_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def extract_preferences_from_text(self, text: str) -> list[tuple[str, str]]:
        """Extract normalized (category, preference_value) pairs after PII scrubbing."""
        scrubbed = self._redactor.redact_text(text).redacted_text
        extracted: list[tuple[str, str]] = []
        for category, pattern in _PREFERENCE_EXTRACTORS:
            for match in pattern.finditer(scrubbed):
                value = match.group(0).strip()
                if value:
                    extracted.append((category, value))
        return extracted

    def schedule_preference_write(
        self,
        *,
        traveler_id: str,
        category: str,
        preference_value: str,
        confidence: float = 0.9,
    ) -> asyncio.Task[Any] | None:
        """Schedule an asynchronous preference write without blocking the caller."""

        async def _persist() -> None:
            async with self._semaphore:
                try:
                    record, had_pii = await asyncio.to_thread(
                        self._store.upsert_preference,
                        traveler_id=traveler_id,
                        category=category,
                        preference_value=preference_value,
                        confidence=confidence,
                    )
                    self._logger.emit(
                        "memory.preference_persisted",
                        "Background traveler preference persisted",
                        payload={
                            "traveler_id": traveler_id,
                            "category": record.category,
                            "preference_id": record.preference_id,
                            "pii_redacted": had_pii,
                        },
                    )
                except Exception as exc:
                    self._logger.emit(
                        "memory.consolidation_error",
                        f"Background preference write failed: {type(exc).__name__}",
                        payload={"traveler_id": traveler_id, "category": category},
                    )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop (e.g., synchronous unit test): persist safely inline.
            self._store.upsert_preference(
                traveler_id=traveler_id,
                category=category,
                preference_value=preference_value,
                confidence=confidence,
            )
            return None

        return self._track_task(loop.create_task(_persist()))

    def schedule_session_consolidation(
        self,
        *,
        session: Session,
        traveler_id: str | None = None,
    ) -> asyncio.Task[Any] | None:
        """Schedule async background consolidation of a session's events into long-term memory."""
        resolved_traveler_id = traveler_id or session.user_id

        async def _consolidate() -> int:
            async with self._semaphore:
                saved_count = 0
                try:
                    for event in session.events:
                        if event.author != "user" or not event.content or not event.content.parts:
                            continue
                        for part in event.content.parts:
                            if not part.text:
                                continue
                            for category, value in self.extract_preferences_from_text(part.text):
                                await asyncio.to_thread(
                                    self._store.upsert_preference,
                                    traveler_id=resolved_traveler_id,
                                    category=category,
                                    preference_value=value,
                                    confidence=0.85,
                                )
                                saved_count += 1

                    if self._memory_service is not None:
                        await self._memory_service.add_session_to_memory(session)

                    self._logger.emit(
                        "memory.session_consolidated",
                        "Async session memory consolidation completed",
                        payload={
                            "session_id": session.id,
                            "traveler_id": resolved_traveler_id,
                            "preferences_extracted": saved_count,
                        },
                    )
                except Exception as exc:
                    self._logger.emit(
                        "memory.consolidation_error",
                        f"Async session consolidation error: {type(exc).__name__}",
                        payload={"session_id": session.id},
                    )
                return saved_count

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None

        return self._track_task(loop.create_task(_consolidate()))

    async def drain(self, timeout: float = 5.0) -> None:
        """Await all in-flight background memory tasks during tests or graceful shutdown."""
        if not self._tasks:
            return
        await asyncio.wait_for(
            asyncio.gather(*list(self._tasks), return_exceptions=True),
            timeout=timeout,
        )
