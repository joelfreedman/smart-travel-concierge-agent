"""Tests for persistent SQLite sessions, history compaction, and async memory consolidation."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.events.event import Event
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from travel_concierge.memory.async_memory import AsyncMemoryConsolidator
from travel_concierge.memory.compaction import (
    SlidingWindowEventSummarizer,
    build_compaction_config,
    estimate_events_tokens,
)
from travel_concierge.memory.store import PersistentTravelStore


@pytest.mark.asyncio
async def test_sqlite_session_service_persists_across_restarts(tmp_path: Path) -> None:
    """Conversational sessions and state must survive re-instantiating SqliteSessionService."""
    db_file = tmp_path / "persistent_sessions.db"
    svc_one = SqliteSessionService(db_path=str(db_file))
    created = await svc_one.create_session(
        app_name="travel_concierge",
        user_id="traveler_persist",
        state={"trip_budget_usd": 2200.0, "destination": "Lisbon"},
    )
    await svc_one.append_event(
        created,
        Event(
            author="user",
            invocation_id="inv_p1",
            content=types.Content(role="user", parts=[types.Part(text="Plan Lisbon under $2200")]),
        ),
    )

    # Re-open fresh service pointing to the same SQLite database file.
    svc_two = SqliteSessionService(db_path=str(db_file))
    loaded = await svc_two.get_session(
        app_name="travel_concierge",
        user_id="traveler_persist",
        session_id=created.id,
    )
    assert loaded is not None
    assert loaded.state["trip_budget_usd"] == 2200.0
    assert len(loaded.events) == 1
    assert "Lisbon" in loaded.events[0].content.parts[0].text


@pytest.mark.asyncio
async def test_sliding_window_history_compaction_scrubs_pii_and_preserves_facts() -> None:
    """SlidingWindowEventSummarizer produces a compact, PII-free ADK EventCompaction."""
    summarizer = SlidingWindowEventSummarizer(max_summary_chars=900)
    events = [
        Event(
            author="user",
            invocation_id=f"inv_{idx}",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            f"Turn {idx}: Keep trip under $1800 and email me at "
                            "secret.user@example.com regarding APR-1234ABCD."
                        )
                    )
                ],
            ),
        )
        for idx in range(1, 9)
    ]

    assert estimate_events_tokens(events) > 50
    compacted = await summarizer.maybe_summarize_events(events=events)
    assert compacted is not None
    assert compacted.actions.compaction is not None

    summary_text = compacted.actions.compaction.compacted_content.parts[0].text
    assert "[Compacted Conversation Summary]" in summary_text
    assert "$1800" in summary_text
    assert "APR-1234ABCD" in summary_text
    assert "secret.user@example.com" not in summary_text
    assert "[REDACTED_EMAIL]" in summary_text

    cfg = build_compaction_config()
    assert cfg.compaction_interval >= 1
    assert cfg.token_threshold > 0


@pytest.mark.asyncio
async def test_async_memory_consolidator_extracts_and_persists_non_blocking(
    tmp_path: Path,
) -> None:
    """AsyncMemoryConsolidator dispatches background tasks without blocking the caller."""
    store = PersistentTravelStore(tmp_path / "async_mem.db")
    consolidator = AsyncMemoryConsolidator(store=store)
    session_service = SqliteSessionService(db_path=str(tmp_path / "async_sessions.db"))
    session = await session_service.create_session(
        app_name="travel_concierge", user_id="traveler_async"
    )
    await session_service.append_event(
        session,
        Event(
            author="user",
            invocation_id="inv_async_1",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            "I prefer window seats and I am vegetarian. "
                            "Please stay near transit! My phone is 512-555-0144."
                        )
                    )
                ],
            ),
        ),
    )

    task = consolidator.schedule_session_consolidation(session=session)
    assert task is not None
    await consolidator.drain()

    prefs = store.list_preferences("traveler_async")
    categories = {p.category for p in prefs}
    assert {"seating", "dietary", "lodging"}.issubset(categories)
    assert all("512-555-0144" not in p.preference_value for p in prefs)
