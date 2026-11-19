"""Schema-validated, richly documented ADK tools with guided error handling and HITL gates.

Design principles:
- Specific action-verb tool names (e.g., ``search_budget_flights``,
  ``calculate_trip_budget_breakdown``, ``execute_confirmed_booking``).
- Comprehensive Google-style docstrings explaining purpose, arguments, returns,
  and error recovery paths so the LLM calls tools accurately.
- Strict Pydantic v2 input/output validation via ``travel_concierge.schemas``.
- Guided error handling: returns ``StructuredToolError`` dictionaries with
  ``recovery_instructions`` instead of crashing the agent loop.
- Dual-layer fail-closed Human-in-the-Loop confirmation on ``execute_confirmed_booking``.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool, ToolContext
from pydantic import ValidationError

from travel_concierge.config.settings import AppSettings
from travel_concierge.memory.async_memory import AsyncMemoryConsolidator
from travel_concierge.memory.store import PersistentTravelStore
from travel_concierge.schemas import (
    BookingApprovalRequest,
    BookingApprovalResponse,
    BookingExecutionRequest,
    BookingExecutionResponse,
    FlightOption,
    FlightSearchRequest,
    FlightSearchResponse,
    HotelOption,
    HotelSearchRequest,
    HotelSearchResponse,
    StructuredToolError,
    TravelerPreferenceInput,
    TravelerPreferenceRecallResponse,
    TravelerPreferenceSaveResponse,
    TripBudgetRequest,
    TripBudgetResponse,
)
from travel_concierge.tools.catalog import (
    FLIGHT_CATALOG,
    HOTEL_CATALOG,
    SUPPORTED_AIRPORTS,
    lookup_flight_by_id,
    lookup_hotel_by_id,
)


def _validation_error_payload(
    error_code: str,
    exc: ValidationError,
    recovery_instructions: str,
    suggested_next_tool: str | None = None,
) -> dict[str, Any]:
    """Build a concise, LLM-friendly recovery response from a Pydantic ValidationError."""
    issues = [
        {
            "field": ".".join(str(part) for part in err.get("loc", ())),
            "reason": err.get("msg", "Invalid parameter value."),
        }
        for err in exc.errors()
    ]
    return StructuredToolError(
        error_code=error_code,
        message="One or more tool arguments failed strict JSON schema validation.",
        recovery_instructions=recovery_instructions,
        suggested_next_tool=suggested_next_tool,
        retryable=True,
        details={"validation_issues": issues},
    ).model_dump()


class TravelConciergeToolset:
    """Stateful toolset binding persistent SQLite storage and async memory to ADK tools."""

    def __init__(
        self,
        *,
        store: PersistentTravelStore,
        memory_consolidator: AsyncMemoryConsolidator,
        settings: AppSettings | None = None,
    ) -> None:
        self._store = store
        self._memory_consolidator = memory_consolidator
        self._settings = settings or AppSettings.from_env()

    # ------------------------------------------------------------------
    # Tool 1: Search Budget Flights
    # ------------------------------------------------------------------

    def search_budget_flights(
        self,
        origin_iata: str,
        destination_iata: str,
        departure_date: str,
        return_date: str,
        max_price_per_person_usd: float,
        travelers: int = 1,
        cabin_class: str = "economy",
        nonstop_only: bool = False,
    ) -> dict[str, Any]:
        """Search round-trip flights that fit within a traveler's per-person airfare ceiling.

        Use this tool during trip planning to retrieve concrete flight quotes with
        exact ``flight_id`` identifiers required before budget modeling or booking holds.

        Args:
            origin_iata: 3-letter uppercase IATA departure airport code (e.g., ``AUS``, ``JFK``).
            destination_iata: 3-letter uppercase IATA arrival airport code (e.g., ``NRT``, ``LIS``).
            departure_date: Outbound travel date in ISO-8601 ``YYYY-MM-DD`` format.
            return_date: Return travel date in ISO-8601 ``YYYY-MM-DD`` format (must be after departure).
            max_price_per_person_usd: Maximum acceptable round-trip airfare per traveler in USD.
            travelers: Number of passengers between 1 and 9 (default: 1).
            cabin_class: Cabin tier: ``economy``, ``premium_economy``, or ``business`` (default: ``economy``).
            nonstop_only: Set ``True`` to exclude flights with layovers (default: ``False``).

        Returns:
            A schema-validated ``FlightSearchResponse`` dict with matching flight options
            sorted by lowest price, or a ``StructuredToolError`` dict with recovery steps
            if airport codes are invalid or no flights fit the airfare ceiling.
        """
        try:
            req = FlightSearchRequest(
                origin_iata=origin_iata,
                destination_iata=destination_iata,
                departure_date=departure_date,  # type: ignore[arg-type]
                return_date=return_date,  # type: ignore[arg-type]
                travelers=travelers,
                max_price_per_person_usd=max_price_per_person_usd,
                cabin_class=cabin_class,  # type: ignore[arg-type]
                nonstop_only=nonstop_only,
            )
        except ValidationError as exc:
            return _validation_error_payload(
                error_code="INVALID_FLIGHT_SEARCH_ARGS",
                exc=exc,
                recovery_instructions=(
                    "Fix invalid airport IATA codes (must be 3-letter codes from "
                    f"{sorted(SUPPORTED_AIRPORTS.keys())}), ensure return_date is strictly after "
                    "departure_date, and retry search_budget_flights."
                ),
                suggested_next_tool="search_budget_flights",
            )

        if req.origin_iata not in SUPPORTED_AIRPORTS or req.destination_iata not in SUPPORTED_AIRPORTS:
            return StructuredToolError(
                error_code="UNSUPPORTED_AIRPORT_CODE",
                message=(
                    f"Route {req.origin_iata}->{req.destination_iata} references an unsupported airport."
                ),
                recovery_instructions=(
                    "Ask the traveler to choose from supported major hub airports "
                    f"({', '.join(f'{code} ({city})' for code, city in sorted(SUPPORTED_AIRPORTS.items()))}) "
                    "and retry search_budget_flights."
                ),
                suggested_next_tool="search_budget_flights",
                retryable=True,
                details={"supported_airports": SUPPORTED_AIRPORTS},
            ).model_dump()

        route_matches = [
            f
            for f in FLIGHT_CATALOG
            if f.origin_iata == req.origin_iata
            and f.destination_iata == req.destination_iata
            and f.cabin_class == req.cabin_class
            and (not req.nonstop_only or f.stops == 0)
        ]

        if not route_matches:
            return StructuredToolError(
                error_code="NO_ROUTE_MATCH",
                message=(
                    f"No {req.cabin_class} flights found from {req.origin_iata} to "
                    f"{req.destination_iata} with nonstop_only={req.nonstop_only}."
                ),
                recovery_instructions=(
                    "Retry search_budget_flights with nonstop_only=False or cabin_class='economy', "
                    "or suggest an alternate departure hub such as SFO, JFK, or AUS."
                ),
                suggested_next_tool="search_budget_flights",
                retryable=True,
            ).model_dump()

        within_cap = sorted(
            [f for f in route_matches if f.price_per_person_usd <= req.max_price_per_person_usd],
            key=lambda item: item.price_per_person_usd,
        )

        if not within_cap:
            cheapest = min(route_matches, key=lambda item: item.price_per_person_usd)
            return StructuredToolError(
                error_code="NO_FLIGHTS_WITHIN_BUDGET",
                message=(
                    f"Lowest available {req.cabin_class} fare on {req.origin_iata}->{req.destination_iata} "
                    f"is ${cheapest.price_per_person_usd:.2f}/person, exceeding the "
                    f"${req.max_price_per_person_usd:.2f}/person cap."
                ),
                recovery_instructions=(
                    f"Either increase max_price_per_person_usd to at least "
                    f"${cheapest.price_per_person_usd:.2f} by reducing lodging spend, switch "
                    "nonstop_only to False, or propose a closer destination such as DEN or MEX."
                ),
                suggested_next_tool="search_budget_flights",
                retryable=True,
                details={"lowest_available_per_person_usd": cheapest.price_per_person_usd},
            ).model_dump()

        options = [
            FlightOption(
                flight_id=f.flight_id,
                airline=f.airline,
                origin_iata=f.origin_iata,
                destination_iata=f.destination_iata,
                departure_date=req.departure_date.isoformat(),
                return_date=req.return_date.isoformat(),
                cabin_class=f.cabin_class,  # type: ignore[arg-type]
                stops=f.stops,
                duration_hours=f.duration_hours,
                price_per_person_usd=f.price_per_person_usd,
                total_price_usd=round(f.price_per_person_usd * req.travelers, 2),
                refundable=f.refundable,
            )
            for f in within_cap
        ]

        return FlightSearchResponse(
            origin_iata=req.origin_iata,
            destination_iata=req.destination_iata,
            travelers=req.travelers,
            budget_cap_per_person_usd=req.max_price_per_person_usd,
            flights=options,
            lowest_total_usd=options[0].total_price_usd,
        ).model_dump()

    # ------------------------------------------------------------------
    # Tool 2: Search Hotels Within Budget
    # ------------------------------------------------------------------

    def search_hotels_within_budget(
        self,
        destination_city: str,
        check_in_date: str,
        check_out_date: str,
        max_nightly_rate_usd: float,
        guests: int = 1,
        preferred_amenities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search hotels in the destination city filtered by nightly rate cap and amenities.

        Returns exact ``hotel_id`` identifiers and tax-inclusive stay totals needed for
        ``calculate_trip_budget_breakdown`` and ``request_booking_approval``.

        Args:
            destination_city: City name (e.g., ``Tokyo``, ``Lisbon``, ``Denver``, ``Paris``).
            check_in_date: Hotel check-in date in ISO-8601 ``YYYY-MM-DD`` format.
            check_out_date: Hotel check-out date in ISO-8601 ``YYYY-MM-DD`` format.
            max_nightly_rate_usd: Upper limit for nightly room price in USD before taxes.
            guests: Number of guests between 1 and 9 (default: 1).
            preferred_amenities: Optional list of desired amenities (e.g., ``["wifi", "transit"]``).

        Returns:
            A ``HotelSearchResponse`` dict ranked by amenity match and nightly value,
            or a ``StructuredToolError`` with actionable recovery steps.
        """
        try:
            req = HotelSearchRequest(
                destination_city=destination_city,
                check_in_date=check_in_date,  # type: ignore[arg-type]
                check_out_date=check_out_date,  # type: ignore[arg-type]
                max_nightly_rate_usd=max_nightly_rate_usd,
                guests=guests,
                preferred_amenities=preferred_amenities or [],
            )
        except ValidationError as exc:
            return _validation_error_payload(
                error_code="INVALID_HOTEL_SEARCH_ARGS",
                exc=exc,
                recovery_instructions=(
                    "Ensure check_out_date is at least 1 night after check_in_date and "
                    "max_nightly_rate_usd is a positive USD amount, then retry search_hotels_within_budget."
                ),
                suggested_next_tool="search_hotels_within_budget",
            )

        nights = (req.check_out_date - req.check_in_date).days
        city_key = req.destination_city.strip().lower()
        city_hotels = [
            h
            for h in HOTEL_CATALOG
            if h.city.lower() == city_key or h.iata_code.lower() == city_key
        ]

        if not city_hotels:
            supported_cities = sorted({h.city for h in HOTEL_CATALOG})
            return StructuredToolError(
                error_code="CITY_NOT_IN_CATALOG",
                message=f"No partner hotels are currently indexed for '{req.destination_city}'.",
                recovery_instructions=(
                    f"Offer supported destinations ({', '.join(supported_cities)}) and ask the "
                    "traveler which city they prefer before re-calling search_hotels_within_budget."
                ),
                suggested_next_tool="search_hotels_within_budget",
                retryable=True,
                details={"supported_cities": supported_cities},
            ).model_dump()

        affordable = [h for h in city_hotels if h.nightly_rate_usd <= req.max_nightly_rate_usd]
        if not affordable:
            cheapest = min(city_hotels, key=lambda item: item.nightly_rate_usd)
            return StructuredToolError(
                error_code="NO_HOTELS_WITHIN_BUDGET",
                message=(
                    f"Cheapest hotel in {cheapest.city} is ${cheapest.nightly_rate_usd:.2f}/night, "
                    f"above your ${req.max_nightly_rate_usd:.2f}/night cap."
                ),
                recovery_instructions=(
                    f"Retry search_hotels_within_budget with max_nightly_rate_usd >= "
                    f"{cheapest.nightly_rate_usd:.2f} (and shorten the trip by 1 night if needed "
                    "to stay inside the total trip budget)."
                ),
                suggested_next_tool="search_hotels_within_budget",
                retryable=True,
                details={"lowest_available_nightly_usd": cheapest.nightly_rate_usd},
            ).model_dump()

        desired = {a.strip().lower() for a in req.preferred_amenities}

        def _rank_key(hotel: Any) -> tuple[int, float]:
            overlap = len(desired.intersection(hotel.amenities))
            return (-overlap, hotel.nightly_rate_usd)

        ranked = sorted(affordable, key=_rank_key)
        options = []
        for hotel in ranked:
            base = round(hotel.nightly_rate_usd * nights, 2)
            taxes = round(base * hotel.tax_rate, 2)
            options.append(
                HotelOption(
                    hotel_id=hotel.hotel_id,
                    name=hotel.name,
                    city=hotel.city,
                    neighborhood=hotel.neighborhood,
                    nightly_rate_usd=hotel.nightly_rate_usd,
                    nights=nights,
                    estimated_taxes_usd=taxes,
                    total_stay_usd=round(base + taxes, 2),
                    rating=hotel.rating,
                    amenities=list(hotel.amenities),
                    free_cancellation=hotel.free_cancellation,
                )
            )

        return HotelSearchResponse(
            destination_city=ranked[0].city,
            nights=nights,
            hotels=options,
        ).model_dump()

    # ------------------------------------------------------------------
    # Tool 3: Calculate Trip Budget Breakdown
    # ------------------------------------------------------------------

    def calculate_trip_budget_breakdown(
        self,
        total_budget_usd: float,
        flight_total_usd: float,
        hotel_total_usd: float,
        nights: int,
        travelers: int = 1,
        daily_food_and_activities_per_person_usd: float = 85.0,
        contingency_percent: float = 10.0,
    ) -> dict[str, Any]:
        """Compute deterministic end-to-end trip cost math and check hard budget compliance.

        Always call this tool before recommending a final itinerary or creating a
        booking approval hold so all math is deterministic rather than LLM-estimated.

        Args:
            total_budget_usd: Traveler's hard upper budget ceiling in USD.
            flight_total_usd: Combined airfare total in USD from ``search_budget_flights``.
            hotel_total_usd: Combined tax-inclusive lodging total from ``search_hotels_within_budget``.
            nights: Number of nights at the destination.
            travelers: Number of travelers (default: 1).
            daily_food_and_activities_per_person_usd: Daily allowance for meals, transit, and tickets.
            contingency_percent: Percentage emergency reserve (default: 10.0%).

        Returns:
            A ``TripBudgetResponse`` dict showing line-item costs, ``within_budget`` boolean,
            remaining or excess dollar delta, and concrete savings recommendations if over budget.
        """
        try:
            req = TripBudgetRequest(
                total_budget_usd=total_budget_usd,
                flight_total_usd=flight_total_usd,
                hotel_total_usd=hotel_total_usd,
                nights=nights,
                travelers=travelers,
                daily_food_and_activities_per_person_usd=daily_food_and_activities_per_person_usd,
                contingency_percent=contingency_percent,
            )
        except ValidationError as exc:
            return _validation_error_payload(
                error_code="INVALID_BUDGET_INPUTS",
                exc=exc,
                recovery_instructions=(
                    "Ensure total_budget_usd, flight_total_usd, hotel_total_usd, and nights "
                    "are valid positive numbers matching quoted totals, then retry."
                ),
                suggested_next_tool="calculate_trip_budget_breakdown",
            )

        trip_days = req.nights + 1
        food_and_activities = round(
            trip_days * req.travelers * req.daily_food_and_activities_per_person_usd, 2
        )
        subtotal = round(req.flight_total_usd + req.hotel_total_usd + food_and_activities, 2)
        contingency = round(subtotal * (req.contingency_percent / 100.0), 2)
        projected_total = round(subtotal + contingency, 2)

        delta = round(req.total_budget_usd - projected_total, 2)
        within_budget = delta >= 0.0
        utilization = round((projected_total / req.total_budget_usd) * 100.0, 1)

        recommendations: list[str] = []
        if not within_budget:
            over_by = abs(delta)
            recommendations.append(
                f"Itinerary exceeds hard ceiling by ${over_by:,.2f}. Do NOT proceed to booking."
            )
            if req.flight_total_usd > req.total_budget_usd * 0.45:
                recommendations.append(
                    "Airfare exceeds 45% of total budget: allow 1-stop economy flights or alternate hub."
                )
            if req.hotel_total_usd > req.total_budget_usd * 0.40:
                recommendations.append(
                    "Lodging exceeds 40% of total budget: lower max_nightly_rate_usd or reduce stay by 1 night."
                )
            if req.daily_food_and_activities_per_person_usd > 70.0:
                recommendations.append(
                    "Trim daily food & activities estimate toward $65/person/day using transit passes."
                )
        else:
            recommendations.append(
                f"Plan complies with budget ceiling with ${delta:,.2f} headroom remaining."
            )

        return TripBudgetResponse(
            within_budget=within_budget,
            total_budget_usd=round(req.total_budget_usd, 2),
            flight_total_usd=round(req.flight_total_usd, 2),
            hotel_total_usd=round(req.hotel_total_usd, 2),
            food_and_activities_total_usd=food_and_activities,
            contingency_usd=contingency,
            projected_total_usd=projected_total,
            remaining_budget_usd=max(0.0, delta),
            over_budget_by_usd=max(0.0, -delta),
            utilization_percent=utilization,
            savings_recommendations=recommendations,
        ).model_dump()

    # ------------------------------------------------------------------
    # Tool 4: Save Traveler Preference (Async Background Write + PII Scrub)
    # ------------------------------------------------------------------

    def save_traveler_preference(
        self,
        traveler_id: str,
        category: str,
        preference_value: str,
        confidence: float = 0.9,
    ) -> dict[str, Any]:
        """Persist a durable traveler preference asynchronously after active PII redaction.

        Call this tool whenever the user reveals a recurring preference (e.g., window
        seat, vegetarian diet, transit-accessible hotels, nonstop preference). Any
        accidental PII (email, phone, passport, card number) is scrubbed prior to storage.

        Args:
            traveler_id: Pseudonymous traveler identifier (alphanumeric, ``_``, ``-``, ``.``).
            category: Preference category: ``seating``, ``airline``, ``lodging``, ``dietary``,
                ``budget_style``, ``destination_interest``, or ``accessibility``.
            preference_value: Short description of the preference (2-240 characters).
            confidence: Confidence score between 0.1 and 1.0 (default: 0.9).

        Returns:
            A ``TravelerPreferenceSaveResponse`` confirming scrubbed text and async dispatch.
        """
        try:
            req = TravelerPreferenceInput(
                traveler_id=traveler_id,
                category=category,  # type: ignore[arg-type]
                preference_value=preference_value,
                confidence=confidence,
            )
        except ValidationError as exc:
            return _validation_error_payload(
                error_code="INVALID_PREFERENCE_INPUT",
                exc=exc,
                recovery_instructions=(
                    "Use a pseudonymous traveler_id (no email addresses or spaces) and a valid "
                    "category (seating, airline, lodging, dietary, budget_style, "
                    "destination_interest, accessibility), then retry save_traveler_preference."
                ),
                suggested_next_tool="save_traveler_preference",
            )

        redaction = self._store.redactor.redact_text(req.preference_value)
        task = self._memory_consolidator.schedule_preference_write(
            traveler_id=req.traveler_id,
            category=req.category,
            preference_value=redaction.redacted_text,
            confidence=req.confidence,
        )
        return TravelerPreferenceSaveResponse(
            traveler_id=req.traveler_id,
            category=req.category,
            stored_value=redaction.redacted_text,
            pii_redacted=redaction.had_pii,
            async_task_dispatched=task is not None,
        ).model_dump()

    # ------------------------------------------------------------------
    # Tool 5: Recall Traveler Preferences (Persistent SQLite Read)
    # ------------------------------------------------------------------

    def recall_traveler_preferences(
        self,
        traveler_id: str,
        category: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Recall persisted traveler preferences from long-term storage to personalize plans.

        Call this at the start of trip intake so previously learned seating, dietary,
        lodging, and budget preferences automatically inform search filters.

        Args:
            traveler_id: Pseudonymous traveler identifier.
            category: Optional preference category filter (e.g., ``lodging`` or ``dietary``).
            query: Optional free-text keyword filter (e.g., ``transit`` or ``window``).

        Returns:
            A ``TravelerPreferenceRecallResponse`` listing matching persisted preferences.
        """
        if not traveler_id or len(traveler_id.strip()) < 2:
            return StructuredToolError(
                error_code="INVALID_TRAVELER_ID",
                message="traveler_id must be at least 2 characters.",
                recovery_instructions="Pass the active session user_id or pseudonymous traveler_id.",
                suggested_next_tool="recall_traveler_preferences",
            ).model_dump()

        records = self._store.list_preferences(
            traveler_id=traveler_id.strip(),
            category=category.strip() if category else None,
            query=query.strip() if query else None,
        )
        return TravelerPreferenceRecallResponse(
            traveler_id=traveler_id.strip(),
            preferences=records,
        ).model_dump()

    # ------------------------------------------------------------------
    # Tool 6: Request Human-in-the-Loop Booking Approval Hold
    # ------------------------------------------------------------------

    def request_booking_approval(
        self,
        traveler_id: str,
        flight_id: str,
        hotel_id: str,
        departure_date: str,
        return_date: str,
        travelers: int,
        total_cost_usd: float,
        trip_budget_usd: float,
        summary: str,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Create a pending Human-in-the-Loop (HITL) booking hold awaiting explicit user sign-off.

        NEVER execute a booking directly without first calling this tool to lock an
        immutable quote fingerprint, verify budget compliance, and present the exact
        charge amount to the human traveler for explicit confirmation.

        Args:
            traveler_id: Pseudonymous traveler identifier owning this reservation hold.
            flight_id: Exact ``flight_id`` obtained from ``search_budget_flights``.
            hotel_id: Exact ``hotel_id`` obtained from ``search_hotels_within_budget``.
            departure_date: Outbound departure date (``YYYY-MM-DD``).
            return_date: Return date (``YYYY-MM-DD``).
            travelers: Number of passengers (1-9).
            total_cost_usd: Combined flight + hotel reservation total in USD.
            trip_budget_usd: Hard trip budget ceiling in USD (must be >= total_cost_usd).
            summary: Human-readable summary of flight, hotel, dates, and cancellation terms.
            tool_context: Injected by ADK runtime for persisting pending approval metadata in session state.

        Returns:
            A ``BookingApprovalResponse`` with ``status='awaiting_human_approval'``, an
            ``approval_id``, expiration timestamp, and the prompt to present to the user.
        """
        try:
            req = BookingApprovalRequest(
                traveler_id=traveler_id,
                flight_id=flight_id,
                hotel_id=hotel_id,
                departure_date=departure_date,  # type: ignore[arg-type]
                return_date=return_date,  # type: ignore[arg-type]
                travelers=travelers,
                total_cost_usd=total_cost_usd,
                trip_budget_usd=trip_budget_usd,
                summary=summary,
            )
        except ValidationError as exc:
            return _validation_error_payload(
                error_code="INVALID_BOOKING_APPROVAL_REQUEST",
                exc=exc,
                recovery_instructions=(
                    "Ensure flight_id/hotel_id came from catalog search tools and total_cost_usd "
                    "does not exceed trip_budget_usd. Recalculate budget before retrying."
                ),
                suggested_next_tool="calculate_trip_budget_breakdown",
            )

        if lookup_flight_by_id(req.flight_id) is None or lookup_hotel_by_id(req.hotel_id) is None:
            return StructuredToolError(
                error_code="UNVERIFIED_INVENTORY_ID",
                message="flight_id or hotel_id does not exist in verified inventory.",
                recovery_instructions=(
                    "Call search_budget_flights and search_hotels_within_budget first and pass "
                    "the exact returned flight_id and hotel_id values."
                ),
                suggested_next_tool="search_budget_flights",
                retryable=True,
            ).model_dump()

        created = self._store.create_booking_approval(
            traveler_id=req.traveler_id,
            flight_id=req.flight_id,
            hotel_id=req.hotel_id,
            departure_date=req.departure_date.isoformat(),
            return_date=req.return_date.isoformat(),
            travelers=req.travelers,
            total_cost_usd=req.total_cost_usd,
            trip_budget_usd=req.trip_budget_usd,
            summary=req.summary,
            ttl_seconds=self._settings.approval_ttl_seconds,
        )

        if tool_context is not None:
            tool_context.state["booking:pending_approval_id"] = created["approval_id"]
            tool_context.state["booking:pending_total_usd"] = created["total_cost_usd"]

        human_prompt = (
            f"HUMAN APPROVAL REQUIRED ({created['approval_id']}): "
            f"Approve booking {req.flight_id} + {req.hotel_id} for {req.travelers} traveler(s) "
            f"({req.departure_date.isoformat()} to {req.return_date.isoformat()}) "
            f"at ${created['total_cost_usd']:,.2f} USD?"
        )

        return BookingApprovalResponse(
            approval_id=created["approval_id"],
            quote_fingerprint=created["quote_fingerprint"],
            total_cost_usd=created["total_cost_usd"],
            expires_at=created["expires_at"],
            human_prompt=human_prompt,
            next_step_instructions=(
                "STOP and present human_prompt to the traveler. Do NOT call "
                "execute_confirmed_booking until the human explicitly approves this approval_id."
            ),
        ).model_dump()

    # ------------------------------------------------------------------
    # Tool 7: Execute Confirmed Booking (Dual-Layer Fail-Closed HITL Gate)
    # ------------------------------------------------------------------

    def execute_confirmed_booking(
        self,
        approval_id: str,
        traveler_id: str,
        expected_total_cost_usd: float,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Finalize a travel reservation ONLY after verified Human-in-the-Loop approval.

        Protected by a dual-layer fail-closed gate:
        1. ADK native ``FunctionTool(..., require_confirmation=True)`` +
           ``tool_context.request_confirmation(...)`` halts execution until the human
           confirms via the ADK confirmation protocol.
        2. Persistent SQLite verification confirms ``approval_id`` exists, belongs to
           ``traveler_id``, matches ``expected_total_cost_usd``, is unexpired, has
           explicit human ``approved`` status, and has not already been consumed.

        Args:
            approval_id: Pending approval ID returned by ``request_booking_approval``.
            traveler_id: Pseudonymous traveler identifier matching the hold.
            expected_total_cost_usd: Exact USD total approved by the traveler.
            tool_context: ADK runtime ``ToolContext`` carrying ``tool_confirmation`` state.

        Returns:
            A ``BookingExecutionResponse`` with a unique ``confirmation_code`` upon verified
            approval, or a ``StructuredToolError`` explaining why execution was blocked.
        """
        try:
            req = BookingExecutionRequest(
                approval_id=approval_id,
                traveler_id=traveler_id,
                expected_total_cost_usd=expected_total_cost_usd,
            )
        except ValidationError as exc:
            return _validation_error_payload(
                error_code="INVALID_BOOKING_EXECUTION_ARGS",
                exc=exc,
                recovery_instructions=(
                    "Pass the exact APR-... approval_id from request_booking_approval along "
                    "with the matching traveler_id and expected_total_cost_usd."
                ),
                suggested_next_tool="request_booking_approval",
            )

        # Layer 1: ADK native ToolConfirmation handshake when running inside ADK ToolContext.
        if tool_context is not None:
            confirmation = tool_context.tool_confirmation
            if confirmation is None:
                # Explicitly pause execution and emit ADK HITL confirmation request.
                tool_context.request_confirmation(
                    hint=(
                        f"Confirm final booking execution for {req.approval_id} "
                        f"(${req.expected_total_cost_usd:,.2f} USD)?"
                    ),
                    payload={
                        "approval_id": req.approval_id,
                        "traveler_id": req.traveler_id,
                        "expected_total_cost_usd": round(req.expected_total_cost_usd, 2),
                    },
                )
                return {
                    "status": "awaiting_human_confirmation",
                    "approval_id": req.approval_id,
                    "message": (
                        "Execution paused at ADK Human-in-the-Loop confirmation hook. "
                        "Awaiting explicit human approval before charging or booking."
                    ),
                }

            if not confirmation.confirmed:
                # Record human rejection if hold is still pending.
                existing = self._store.get_booking_approval(req.approval_id)
                if existing and existing.get("status") == "pending":
                    self._store.record_human_decision(
                        approval_id=req.approval_id,
                        approved=False,
                        reviewer_id=req.traveler_id,
                    )
                return StructuredToolError(
                    error_code="HUMAN_REJECTED_BOOKING",
                    message=f"Human traveler rejected booking hold {req.approval_id}.",
                    recovery_instructions=(
                        "Acknowledge the cancellation politely and ask if the traveler wants "
                        "to adjust dates, destination, or budget instead."
                    ),
                    retryable=False,
                ).model_dump()

            # Human confirmed via ADK ToolConfirmation UI/CLI: record decision in SQLite.
            existing = self._store.get_booking_approval(req.approval_id)
            if existing and existing.get("status") == "pending":
                self._store.record_human_decision(
                    approval_id=req.approval_id,
                    approved=True,
                    reviewer_id=req.traveler_id,
                )

        # Layer 2: Persistent single-use atomic approval verification (fail-closed).
        ok, reason, record = self._store.consume_approved_booking(
            approval_id=req.approval_id,
            traveler_id=req.traveler_id,
            expected_total_cost_usd=req.expected_total_cost_usd,
        )
        if not ok or record is None:
            recovery_map = {
                "APPROVAL_NOT_FOUND": (
                    "Create a valid approval hold via request_booking_approval before booking."
                ),
                "HUMAN_APPROVAL_REQUIRED": (
                    "STOP immediately. Present the approval details to the human traveler and "
                    "wait for explicit approval before attempting execute_confirmed_booking again."
                ),
                "AMOUNT_MISMATCH": (
                    "The requested charge does not match the human-approved quote. Create a new "
                    "hold via request_booking_approval with the updated total."
                ),
                "TRAVELER_MISMATCH": (
                    "Approval ID belongs to a different traveler. Do not cross-use approvals."
                ),
                "APPROVAL_EXPIRED": (
                    "The approval hold expired. Re-verify prices and call request_booking_approval."
                ),
                "APPROVAL_ALREADY_CONSUMED": (
                    "This approval was already executed (single-use protection). Share the "
                    "existing confirmation code instead of double-booking."
                ),
            }
            return StructuredToolError(
                error_code=reason,
                message=f"Booking execution blocked by fail-closed HITL gate ({reason}).",
                recovery_instructions=recovery_map.get(
                    reason,
                    "Verify human approval state and create a fresh hold if needed.",
                ),
                suggested_next_tool=(
                    "request_booking_approval"
                    if reason in {"APPROVAL_NOT_FOUND", "APPROVAL_EXPIRED", "AMOUNT_MISMATCH"}
                    else None
                ),
                retryable=reason != "APPROVAL_ALREADY_CONSUMED",
            ).model_dump()

        if tool_context is not None:
            tool_context.state["booking:last_confirmed_code"] = record["confirmation_code"]

        return BookingExecutionResponse(
            confirmation_code=str(record["confirmation_code"]),
            approval_id=str(record["approval_id"]),
            flight_id=str(record["flight_id"]),
            hotel_id=str(record["hotel_id"]),
            total_charged_usd=float(record["total_cost_usd"]),
            human_approved_by=str(record["approved_by"]),
            approved_at=str(record["approved_at"]),
        ).model_dump()

    def build_booking_function_tool(self) -> FunctionTool:
        """Wrap ``execute_confirmed_booking`` with ADK's native ``require_confirmation=True``."""
        return FunctionTool(
            func=self.execute_confirmed_booking,
            require_confirmation=True,
        )
