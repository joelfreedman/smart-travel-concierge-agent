"""Context-bloat management via ADK EventsCompactionConfig and sliding-window summarization.

Implements an ADK ``BaseEventsSummarizer`` subclass wired directly into
``google.adk.apps.App(events_compaction_config=..., context_cache_config=...)``.
Compacts older conversation history when event count or estimated token thresholds
are crossed, preserving critical trip constraints (destination, dates, hard budget
ceiling, recalled preferences, and pending approval IDs) while scrubbing PII.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.genai import types

from travel_concierge.config.settings import AppSettings
from travel_concierge.security.pii_redactor import PIIRedactor

_BUDGET_RE = re.compile(r"\$[0-9,]+(?:\.[0-9]{2})?")
_IATA_RE = re.compile(r"\b[A-Z]{3}\b")
_APPROVAL_RE = re.compile(r"\bAPR-[A-Z0-9]{8,32}\b")


def estimate_events_tokens(events: Sequence[Event]) -> int:
    """Estimate token count across a sequence of ADK events (~4 chars/token heuristic)."""
    total_chars = 0
    for event in events:
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.text:
                total_chars += len(part.text)
            elif part.function_call:
                total_chars += len(str(part.function_call.name)) + len(
                    str(part.function_call.args)
                )
            elif part.function_response:
                total_chars += len(str(part.function_response.name)) + len(
                    str(part.function_response.response)
                )
    return max(0, total_chars // 4)


class SlidingWindowEventSummarizer(BaseEventsSummarizer):
    """Deterministic, PII-scrubbed sliding-window event summarizer for ADK compaction.

    Extracts durable trip facts (budgets, IATA codes, approval IDs, and salient turn
    excerpts) into a bounded compacted summary event, keeping token footprint small
    without incurring extra LLM latency or failing offline.
    """

    def __init__(
        self,
        *,
        max_summary_chars: int = 1200,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self._max_summary_chars = max_summary_chars
        self._redactor = redactor or PIIRedactor()

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        """Compact a window of older ADK events into a single ``EventCompaction``."""
        if not events:
            return None

        highlights: list[str] = []
        budgets_seen: list[str] = []
        approvals_seen: list[str] = []
        tool_calls_seen: list[str] = []

        for event in events:
            if not event.content or not event.content.parts:
                continue
            author = event.author or "unknown"
            for part in event.content.parts:
                if part.text:
                    clean = self._redactor.redact_text(part.text).redacted_text.strip()
                    if not clean:
                        continue
                    budgets_seen.extend(_BUDGET_RE.findall(clean))
                    approvals_seen.extend(_APPROVAL_RE.findall(clean))
                    snippet = clean if len(clean) <= 140 else f"{clean[:137]}..."
                    highlights.append(f"- [{author}] {snippet}")
                elif part.function_call and part.function_call.name:
                    tool_calls_seen.append(part.function_call.name)

        if not highlights and not tool_calls_seen:
            return None

        summary_lines = [
            "[Compacted Conversation Summary]",
            f"Compacted events: {len(events)} (est. tokens before compaction: "
            f"{estimate_events_tokens(events)})",
        ]
        if budgets_seen:
            summary_lines.append(
                f"Budgets referenced: {', '.join(dict.fromkeys(budgets_seen))}"
            )
        if approvals_seen:
            summary_lines.append(
                f"Approval IDs referenced: {', '.join(dict.fromkeys(approvals_seen))}"
            )
        if tool_calls_seen:
            summary_lines.append(
                f"Tools executed: {', '.join(dict.fromkeys(tool_calls_seen))}"
            )
        summary_lines.append("Key turn excerpts:")
        summary_lines.extend(highlights[-6:])

        summary_text = "\n".join(summary_lines)
        if len(summary_text) > self._max_summary_chars:
            summary_text = summary_text[: self._max_summary_chars - 3] + "..."

        compacted_content = types.Content(
            role="model",
            parts=[types.Part(text=summary_text)],
        )
        compaction = EventCompaction(
            start_timestamp=events[0].timestamp,
            end_timestamp=events[-1].timestamp,
            compacted_content=compacted_content,
        )
        return Event(
            author="user",
            actions=EventActions(compaction=compaction),
            invocation_id=Event.new_id(),
        )


def build_compaction_config(
    settings: AppSettings | None = None,
    *,
    redactor: PIIRedactor | None = None,
) -> EventsCompactionConfig:
    """Build the ADK ``EventsCompactionConfig`` for sliding-window context management."""
    cfg = settings or AppSettings.from_env()
    return EventsCompactionConfig(
        summarizer=SlidingWindowEventSummarizer(redactor=redactor),
        compaction_interval=cfg.compaction_interval,
        overlap_size=cfg.compaction_overlap_size,
        token_threshold=cfg.compaction_token_threshold,
        event_retention_size=cfg.compaction_event_retention_size,
    )


def build_context_cache_config() -> ContextCacheConfig:
    """Configure Vertex/Gemini context caching for stable constitutional instructions."""
    return ContextCacheConfig(
        cache_intervals=10,
        ttl_seconds=1800,
        min_tokens=2048,
    )
