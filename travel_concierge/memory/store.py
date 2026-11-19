"""Persistent SQLite storage for sessions, traveler preferences, and HITL booking approvals.

Security & reliability invariants:
- 100% parameterized SQL statements (zero string formatting or interpolation in queries).
- Explicit atomic state transitions for single-use Human-in-the-Loop approval tokens
  to prevent double-booking or parameter tampering.
- Supports switching seamlessly between local SQLite persistence and Vertex AI
  Session / Memory Bank services via ``AGENT_STORAGE_BACKEND``.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

from travel_concierge.config.settings import AppSettings
from travel_concierge.schemas import TravelerPreferenceRecord
from travel_concierge.security.pii_redactor import PIIRedactor


def compute_quote_fingerprint(
    *,
    traveler_id: str,
    flight_id: str,
    hotel_id: str,
    departure_date: str,
    return_date: str,
    travelers: int,
    total_cost_usd: float,
) -> str:
    """Compute a deterministic SHA-256 fingerprint binding a booking quote to its parameters."""
    canonical = json.dumps(
        {
            "traveler_id": traveler_id,
            "flight_id": flight_id,
            "hotel_id": hotel_id,
            "departure_date": departure_date,
            "return_date": return_date,
            "travelers": travelers,
            "total_cost_usd": round(total_cost_usd, 2),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class PersistentTravelStore:
    """SQLite-backed store for traveler preferences and HITL booking approvals."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._redactor = redactor or PIIRedactor()
        self._init_schema()

    @property
    def db_path(self) -> Path:
        """Return resolved SQLite database file path."""
        return self._db_path

    @property
    def redactor(self) -> PIIRedactor:
        """Return the PII redactor protecting persistence writes."""
        return self._redactor

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traveler_preferences (
                    preference_id TEXT PRIMARY KEY,
                    traveler_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(traveler_id, category, preference_value)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_preferences_traveler
                ON traveler_preferences (traveler_id, updated_at DESC);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS booking_approvals (
                    approval_id TEXT PRIMARY KEY,
                    traveler_id TEXT NOT NULL,
                    flight_id TEXT NOT NULL,
                    hotel_id TEXT NOT NULL,
                    departure_date TEXT NOT NULL,
                    return_date TEXT NOT NULL,
                    travelers INTEGER NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    trip_budget_usd REAL NOT NULL,
                    summary TEXT NOT NULL,
                    quote_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    approved_by TEXT,
                    approved_at TEXT,
                    confirmation_code TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------
    # Traveler Preference Persistence (PII-scrubbed before write)
    # ------------------------------------------------------------------

    def upsert_preference(
        self,
        *,
        traveler_id: str,
        category: str,
        preference_value: str,
        confidence: float = 0.9,
    ) -> tuple[TravelerPreferenceRecord, bool]:
        """Scrub PII and persist a traveler preference using parameterized SQL."""
        redaction = self._redactor.redact_text(preference_value)
        clean_value = redaction.redacted_text.strip()
        now_iso = datetime.now(UTC).isoformat()
        pref_id = f"PREF-{secrets.token_hex(6).upper()}"

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT preference_id
                FROM traveler_preferences
                WHERE traveler_id = ? AND category = ? AND preference_value = ?
                """,
                (traveler_id, category, clean_value),
            ).fetchone()

            if existing is not None:
                pref_id = str(existing["preference_id"])
                conn.execute(
                    """
                    UPDATE traveler_preferences
                    SET confidence = ?, updated_at = ?
                    WHERE preference_id = ?
                    """,
                    (confidence, now_iso, pref_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO traveler_preferences (
                        preference_id, traveler_id, category,
                        preference_value, confidence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (pref_id, traveler_id, category, clean_value, confidence, now_iso),
                )

        record = TravelerPreferenceRecord(
            preference_id=pref_id,
            traveler_id=traveler_id,
            category=category,
            preference_value=clean_value,
            confidence=confidence,
            updated_at=now_iso,
        )
        return record, redaction.had_pii

    def list_preferences(
        self,
        traveler_id: str,
        *,
        category: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[TravelerPreferenceRecord]:
        """Retrieve persisted preferences for a traveler with optional filtering."""
        bounded_limit = max(1, min(limit, 100))
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    """
                    SELECT preference_id, traveler_id, category,
                           preference_value, confidence, updated_at
                    FROM traveler_preferences
                    WHERE traveler_id = ? AND category = ?
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (traveler_id, category, bounded_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT preference_id, traveler_id, category,
                           preference_value, confidence, updated_at
                    FROM traveler_preferences
                    WHERE traveler_id = ?
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (traveler_id, bounded_limit),
                ).fetchall()

        records = [
            TravelerPreferenceRecord(
                preference_id=str(row["preference_id"]),
                traveler_id=str(row["traveler_id"]),
                category=str(row["category"]),
                preference_value=str(row["preference_value"]),
                confidence=float(row["confidence"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

        if query:
            tokens = [tok.lower() for tok in query.split() if len(tok) >= 2]
            if tokens:
                records = [
                    rec
                    for rec in records
                    if any(
                        tok in rec.preference_value.lower() or tok in rec.category.lower()
                        for tok in tokens
                    )
                ]
        return records

    # ------------------------------------------------------------------
    # Human-in-the-Loop (HITL) Booking Approval Lifecycle
    # ------------------------------------------------------------------

    def create_booking_approval(
        self,
        *,
        traveler_id: str,
        flight_id: str,
        hotel_id: str,
        departure_date: str,
        return_date: str,
        travelers: int,
        total_cost_usd: float,
        trip_budget_usd: float,
        summary: str,
        ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        """Persist a pending HITL booking approval record awaiting human sign-off."""
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=ttl_seconds)
        approval_id = f"APR-{secrets.token_hex(6).upper()}"
        clean_summary = self._redactor.redact_text(summary).redacted_text
        fingerprint = compute_quote_fingerprint(
            traveler_id=traveler_id,
            flight_id=flight_id,
            hotel_id=hotel_id,
            departure_date=departure_date,
            return_date=return_date,
            travelers=travelers,
            total_cost_usd=total_cost_usd,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO booking_approvals (
                    approval_id, traveler_id, flight_id, hotel_id,
                    departure_date, return_date, travelers,
                    total_cost_usd, trip_budget_usd, summary,
                    quote_fingerprint, status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    traveler_id,
                    flight_id,
                    hotel_id,
                    departure_date,
                    return_date,
                    travelers,
                    round(total_cost_usd, 2),
                    round(trip_budget_usd, 2),
                    clean_summary,
                    fingerprint,
                    "pending",
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )

        return {
            "approval_id": approval_id,
            "quote_fingerprint": fingerprint,
            "total_cost_usd": round(total_cost_usd, 2),
            "expires_at": expires.isoformat(),
            "status": "pending",
        }

    def get_booking_approval(self, approval_id: str) -> dict[str, Any] | None:
        """Fetch an approval record by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM booking_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_human_decision(
        self,
        *,
        approval_id: str,
        approved: bool,
        reviewer_id: str,
    ) -> dict[str, Any]:
        """Record an explicit human approval or rejection for a pending hold.

        Intended to be called by the human UI, CLI confirmation handler, or ADK
        ``ToolConfirmation`` resume step — never unilaterally by the LLM.
        """
        record = self.get_booking_approval(approval_id)
        if record is None:
            raise KeyError(f"Approval record '{approval_id}' does not exist.")

        now = datetime.now(UTC)
        expires_at = datetime.fromisoformat(str(record["expires_at"]))
        if now > expires_at:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE booking_approvals
                    SET status = 'expired'
                    WHERE approval_id = ? AND status = 'pending'
                    """,
                    (approval_id,),
                )
            raise ValueError(f"Approval record '{approval_id}' has expired.")

        new_status = "approved" if approved else "rejected"
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE booking_approvals
                SET status = ?, approved_by = ?, approved_at = ?
                WHERE approval_id = ? AND status = 'pending'
                """,
                (new_status, reviewer_id, now.isoformat(), approval_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"Approval '{approval_id}' cannot transition from status '{record['status']}'."
                )

        updated = self.get_booking_approval(approval_id)
        assert updated is not None
        return updated

    def consume_approved_booking(
        self,
        *,
        approval_id: str,
        traveler_id: str,
        expected_total_cost_usd: float,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Atomically transition an approved hold to ``executed`` (single-use, fail-closed)."""
        record = self.get_booking_approval(approval_id)
        if record is None:
            return False, "APPROVAL_NOT_FOUND", None

        if str(record["traveler_id"]) != traveler_id:
            return False, "TRAVELER_MISMATCH", record

        now = datetime.now(UTC)
        expires_at = datetime.fromisoformat(str(record["expires_at"]))
        if now > expires_at:
            return False, "APPROVAL_EXPIRED", record

        if record["status"] == "executed":
            return False, "APPROVAL_ALREADY_CONSUMED", record

        if record["status"] != "approved":
            return False, "HUMAN_APPROVAL_REQUIRED", record

        if abs(float(record["total_cost_usd"]) - round(expected_total_cost_usd, 2)) > 0.01:
            return False, "AMOUNT_MISMATCH", record

        confirmation_code = f"TRP-{secrets.token_hex(4).upper()}"
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE booking_approvals
                SET status = 'executed', confirmation_code = ?
                WHERE approval_id = ? AND status = 'approved'
                """,
                (confirmation_code, approval_id),
            )
            if cursor.rowcount != 1:
                return False, "CONCURRENT_EXECUTION_BLOCKED", record

        final_record = self.get_booking_approval(approval_id)
        return True, "OK", final_record


# ---------------------------------------------------------------------------
# ADK Session & Memory Service Factories
# ---------------------------------------------------------------------------


def build_session_service(settings: AppSettings | None = None) -> BaseSessionService:
    """Instantiate the configured persistent ADK SessionService."""
    cfg = settings or AppSettings.from_env()
    if cfg.storage_backend == "vertex" and cfg.gcp_project_id and cfg.vertex_agent_engine_id:
        return VertexAiSessionService(
            project=cfg.gcp_project_id,
            location=cfg.gcp_location,
            agent_engine_id=cfg.vertex_agent_engine_id,
        )

    cfg.session_db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSessionService(db_path=str(cfg.session_db_path))


def build_memory_service(settings: AppSettings | None = None) -> BaseMemoryService:
    """Instantiate the configured ADK MemoryService (Vertex Memory Bank or local)."""
    cfg = settings or AppSettings.from_env()
    if cfg.storage_backend == "vertex" and cfg.gcp_project_id and cfg.vertex_agent_engine_id:
        return VertexAiMemoryBankService(
            project=cfg.gcp_project_id,
            location=cfg.gcp_location,
            agent_engine_id=cfg.vertex_agent_engine_id,
        )
    return InMemoryMemoryService()
