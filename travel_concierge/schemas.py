"""Strict Pydantic v2 JSON schemas constraining tool inputs, tool outputs, and agent handoffs.

Every tool validates incoming LLM arguments against an explicit request model
with ``extra="forbid"`` and returns a schema-validated JSON dictionary (either a
typed success payload or a ``StructuredToolError`` containing guided recovery
instructions).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base schema forbidding unknown fields to constrain LLM tool arguments."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Guided Error Schema
# ---------------------------------------------------------------------------


class StructuredToolError(StrictBaseModel):
    """Actionable error response returned to the LLM instead of unhandled crashes."""

    status: Literal["error"] = "error"
    error_code: str = Field(
        description="Machine-readable error code (e.g., INVALID_IATA_CODE, BUDGET_EXCEEDED)."
    )
    message: str = Field(description="Clear explanation of why the tool call could not succeed.")
    recovery_instructions: str = Field(
        description="Step-by-step guidance telling the agent how to recover or ask the user."
    )
    suggested_next_tool: str | None = Field(
        default=None,
        description="Optional recommended follow-up tool to call to resolve the issue.",
    )
    retryable: bool = Field(
        default=True,
        description="Whether retrying with corrected parameters is expected to succeed.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe, PII-scrubbed structured diagnostics for corrective reasoning.",
    )


# ---------------------------------------------------------------------------
# Flight Search Schemas
# ---------------------------------------------------------------------------


class FlightSearchRequest(StrictBaseModel):
    """Validated parameters for searching round-trip budget flights."""

    origin_iata: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="3-letter uppercase IATA origin airport code (e.g., AUS, JFK, SFO).",
    )
    destination_iata: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="3-letter uppercase IATA destination airport code (e.g., NRT, LIS, DEN).",
    )
    departure_date: date = Field(
        description="Outbound departure date in ISO format (YYYY-MM-DD)."
    )
    return_date: date = Field(description="Return flight date in ISO format (YYYY-MM-DD).")
    travelers: int = Field(
        default=1, ge=1, le=9, description="Number of passengers (between 1 and 9)."
    )
    max_price_per_person_usd: float = Field(
        gt=0.0,
        le=25000.0,
        description="Maximum acceptable round-trip airfare per traveler in USD.",
    )
    cabin_class: Literal["economy", "premium_economy", "business"] = Field(
        default="economy",
        description="Requested cabin class.",
    )
    nonstop_only: bool = Field(
        default=False,
        description="If true, only return direct nonstop flight options.",
    )

    @field_validator("origin_iata", "destination_iata", mode="before")
    @classmethod
    def _normalize_iata(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @model_validator(mode="after")
    def _validate_trip_dates_and_route(self) -> FlightSearchRequest:
        if self.origin_iata == self.destination_iata:
            raise ValueError("origin_iata and destination_iata must be different airports.")
        if self.return_date <= self.departure_date:
            raise ValueError("return_date must be strictly after departure_date.")
        if (self.return_date - self.departure_date).days > 60:
            raise ValueError("Trip length cannot exceed 60 days for concierge flight quotes.")
        return self


class FlightOption(StrictBaseModel):
    """Single round-trip flight quote."""

    flight_id: str
    airline: str
    origin_iata: str
    destination_iata: str
    departure_date: str
    return_date: str
    cabin_class: Literal["economy", "premium_economy", "business"]
    stops: int = Field(ge=0, le=3)
    duration_hours: float = Field(gt=0.0)
    price_per_person_usd: float = Field(gt=0.0)
    total_price_usd: float = Field(gt=0.0)
    refundable: bool


class FlightSearchResponse(StrictBaseModel):
    """Schema-constrained flight search result."""

    status: Literal["ok"] = "ok"
    origin_iata: str
    destination_iata: str
    travelers: int
    budget_cap_per_person_usd: float
    flights: list[FlightOption]
    lowest_total_usd: float


# ---------------------------------------------------------------------------
# Hotel Search Schemas
# ---------------------------------------------------------------------------


class HotelSearchRequest(StrictBaseModel):
    """Validated parameters for finding accommodations within a nightly budget."""

    destination_city: str = Field(
        min_length=2,
        max_length=80,
        description="Destination city name (e.g., Tokyo, Lisbon, Denver, Austin).",
    )
    check_in_date: date = Field(description="Hotel check-in date (YYYY-MM-DD).")
    check_out_date: date = Field(description="Hotel check-out date (YYYY-MM-DD).")
    max_nightly_rate_usd: float = Field(
        gt=0.0,
        le=5000.0,
        description="Maximum nightly room rate in USD (before taxes/fees).",
    )
    guests: int = Field(default=1, ge=1, le=9, description="Total number of guests.")
    preferred_amenities: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Optional preferred amenities (e.g., wifi, transit, breakfast, gym).",
    )

    @model_validator(mode="after")
    def _validate_stay_dates(self) -> HotelSearchRequest:
        nights = (self.check_out_date - self.check_in_date).days
        if nights < 1:
            raise ValueError("check_out_date must be at least 1 night after check_in_date.")
        if nights > 60:
            raise ValueError("Hotel stay cannot exceed 60 nights.")
        return self


class HotelOption(StrictBaseModel):
    """Single hotel accommodation quote."""

    hotel_id: str
    name: str
    city: str
    neighborhood: str
    nightly_rate_usd: float = Field(gt=0.0)
    nights: int = Field(ge=1)
    estimated_taxes_usd: float = Field(ge=0.0)
    total_stay_usd: float = Field(gt=0.0)
    rating: float = Field(ge=1.0, le=5.0)
    amenities: list[str]
    free_cancellation: bool


class HotelSearchResponse(StrictBaseModel):
    """Schema-constrained hotel search result."""

    status: Literal["ok"] = "ok"
    destination_city: str
    nights: int
    hotels: list[HotelOption]


# ---------------------------------------------------------------------------
# Trip Budget Calculation Schemas
# ---------------------------------------------------------------------------


class TripBudgetRequest(StrictBaseModel):
    """Validated inputs for total trip cost modeling and budget ceiling enforcement."""

    total_budget_usd: float = Field(
        gt=0.0,
        le=25000.0,
        description="Hard upper trip budget ceiling in USD specified by the traveler.",
    )
    flight_total_usd: float = Field(
        ge=0.0, description="Combined round-trip airfare total in USD for all travelers."
    )
    hotel_total_usd: float = Field(
        ge=0.0, description="Combined hotel total in USD including taxes."
    )
    nights: int = Field(ge=1, le=60, description="Number of trip nights.")
    travelers: int = Field(default=1, ge=1, le=9, description="Number of travelers.")
    daily_food_and_activities_per_person_usd: float = Field(
        default=85.0,
        ge=20.0,
        le=1500.0,
        description="Estimated daily meals, local transit, and activities spend per person.",
    )
    contingency_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=30.0,
        description="Emergency buffer percentage applied to estimated trip costs.",
    )


class TripBudgetResponse(StrictBaseModel):
    """Structured budget analysis with compliance status and optimization levers."""

    status: Literal["ok"] = "ok"
    within_budget: bool
    total_budget_usd: float
    flight_total_usd: float
    hotel_total_usd: float
    food_and_activities_total_usd: float
    contingency_usd: float
    projected_total_usd: float
    remaining_budget_usd: float
    over_budget_by_usd: float
    utilization_percent: float
    savings_recommendations: list[str]


# ---------------------------------------------------------------------------
# Traveler Preference Memory Schemas
# ---------------------------------------------------------------------------


class TravelerPreferenceInput(StrictBaseModel):
    """Validated traveler preference record stored asynchronously after PII scrubbing."""

    traveler_id: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Pseudonymous traveler identifier (never an email, SSN, or passport number).",
    )
    category: Literal[
        "seating",
        "airline",
        "lodging",
        "dietary",
        "budget_style",
        "destination_interest",
        "accessibility",
    ] = Field(description="Normalized traveler preference category.")
    preference_value: str = Field(
        min_length=2,
        max_length=240,
        description="Preference statement (automatically scrubbed for PII before storage).",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Confidence score between 0.1 and 1.0.",
    )


class TravelerPreferenceRecord(StrictBaseModel):
    """Persisted traveler preference returned from memory recall."""

    preference_id: str
    traveler_id: str
    category: str
    preference_value: str
    confidence: float
    updated_at: str


class TravelerPreferenceSaveResponse(StrictBaseModel):
    """Response confirming preference was scrubbed and scheduled for persistence."""

    status: Literal["ok"] = "ok"
    traveler_id: str
    category: str
    stored_value: str
    pii_redacted: bool
    async_task_dispatched: bool


class TravelerPreferenceRecallResponse(StrictBaseModel):
    """Response containing recalled traveler preferences for personalization."""

    status: Literal["ok"] = "ok"
    traveler_id: str
    preferences: list[TravelerPreferenceRecord]


# ---------------------------------------------------------------------------
# Human-in-the-Loop (HITL) Booking Approval & Execution Schemas
# ---------------------------------------------------------------------------


class BookingApprovalRequest(StrictBaseModel):
    """Request to create a pending Human-in-the-Loop booking approval hold."""

    traveler_id: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Pseudonymous traveler identifier owning this booking hold.",
    )
    flight_id: str = Field(
        pattern=r"^FLT-[A-Z0-9-]{4,32}$",
        description="Exact flight_id returned by search_budget_flights.",
    )
    hotel_id: str = Field(
        pattern=r"^HTL-[A-Z0-9-]{4,32}$",
        description="Exact hotel_id returned by search_hotels_within_budget.",
    )
    departure_date: date = Field(description="Trip departure date (YYYY-MM-DD).")
    return_date: date = Field(description="Trip return date (YYYY-MM-DD).")
    travelers: int = Field(default=1, ge=1, le=9)
    total_cost_usd: float = Field(gt=0.0, le=25000.0)
    trip_budget_usd: float = Field(gt=0.0, le=25000.0)
    summary: str = Field(
        min_length=5,
        max_length=400,
        description="Plain-English itinerary summary shown to the human for approval.",
    )

    @model_validator(mode="after")
    def _validate_budget_and_dates(self) -> BookingApprovalRequest:
        if self.return_date <= self.departure_date:
            raise ValueError("return_date must be after departure_date.")
        if self.total_cost_usd > self.trip_budget_usd:
            raise ValueError(
                "total_cost_usd exceeds trip_budget_usd. Revise itinerary before requesting booking approval."
            )
        return self


class BookingApprovalResponse(StrictBaseModel):
    """Pending HITL hold response requiring explicit human confirmation."""

    status: Literal["awaiting_human_approval"] = "awaiting_human_approval"
    approval_id: str
    quote_fingerprint: str
    total_cost_usd: float
    expires_at: str
    human_prompt: str
    next_step_instructions: str


class BookingExecutionRequest(StrictBaseModel):
    """Validated parameters to finalize a human-approved reservation."""

    approval_id: str = Field(
        pattern=r"^APR-[A-Za-z0-9]{8,32}$",
        description="Approval ID returned by request_booking_approval.",
    )
    traveler_id: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Traveler ID that matches the original approval request.",
    )
    expected_total_cost_usd: float = Field(
        gt=0.0,
        le=25000.0,
        description="Exact dollar total approved by the human traveler.",
    )


class BookingExecutionResponse(StrictBaseModel):
    """Confirmed booking receipt returned only after verified human approval."""

    status: Literal["confirmed"] = "confirmed"
    confirmation_code: str
    approval_id: str
    flight_id: str
    hotel_id: str
    total_charged_usd: float
    human_approved_by: str
    approved_at: str


# ---------------------------------------------------------------------------
# Sub-Agent Structured Output Schemas
# ---------------------------------------------------------------------------


class TripIntakeSummary(StrictBaseModel):
    """Structured handoff emitted by the Trip Intake Specialist into session state."""

    origin_iata: str = Field(pattern=r"^[A-Z]{3}$")
    destination_city: str
    destination_iata: str = Field(pattern=r"^[A-Z]{3}$")
    departure_date: str
    return_date: str
    travelers: int = Field(ge=1, le=9)
    total_budget_usd: float = Field(gt=0.0)
    recalled_preferences: list[str] = Field(default_factory=list)
    notes: str = ""
