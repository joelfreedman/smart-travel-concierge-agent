"""Unit tests verifying tool docstrings, naming, strict Pydantic schemas, and guided errors."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_concierge.memory.async_memory import AsyncMemoryConsolidator
from travel_concierge.memory.store import PersistentTravelStore
from travel_concierge.schemas import FlightSearchRequest, HotelSearchRequest
from travel_concierge.tools.travel_tools import TravelConciergeToolset


@pytest.fixture()
def toolset(tmp_path: Path) -> TravelConciergeToolset:
    store = PersistentTravelStore(tmp_path / "tools_test.db")
    consolidator = AsyncMemoryConsolidator(store=store)
    return TravelConciergeToolset(store=store, memory_consolidator=consolidator)


def test_all_tools_have_descriptive_names_and_comprehensive_docstrings(
    toolset: TravelConciergeToolset,
) -> None:
    """Verify every agent tool exposes a specific verb-noun name and parameter docs."""
    tool_functions = [
        toolset.search_budget_flights,
        toolset.search_hotels_within_budget,
        toolset.calculate_trip_budget_breakdown,
        toolset.save_traveler_preference,
        toolset.recall_traveler_preferences,
        toolset.request_booking_approval,
        toolset.execute_confirmed_booking,
    ]
    for fn in tool_functions:
        doc = inspect.getdoc(fn) or ""
        assert "_" in fn.__name__, f"Expected descriptive snake_case name for {fn.__name__}"
        assert "Args:" in doc, f"Missing Args docstring section in {fn.__name__}"
        assert "Returns:" in doc, f"Missing Returns docstring section in {fn.__name__}"
        for param_name in inspect.signature(fn).parameters:
            if param_name == "self":
                continue
            assert param_name in doc, f"Parameter '{param_name}' undocumented in {fn.__name__}"


def test_strict_schemas_forbid_extra_fields_and_invalid_dates() -> None:
    """Strict schemas must reject unknown arguments and reversed dates."""
    with pytest.raises(ValidationError):
        FlightSearchRequest(
            origin_iata="AUS",
            destination_iata="LIS",
            departure_date="2026-10-14",  # type: ignore[arg-type]
            return_date="2026-10-10",  # type: ignore[arg-type]
            max_price_per_person_usd=900.0,
        )

    with pytest.raises(ValidationError):
        HotelSearchRequest(
            destination_city="Lisbon",
            check_in_date="2026-10-10",  # type: ignore[arg-type]
            check_out_date="2026-10-14",  # type: ignore[arg-type]
            max_nightly_rate_usd=180.0,
            unexpected_llm_field="should_fail",  # type: ignore[call-arg]
        )


def test_flight_search_success_and_guided_error_recovery(toolset: TravelConciergeToolset) -> None:
    """Valid searches return sorted quotes; invalid/over-budget queries return guided errors."""
    ok_res = toolset.search_budget_flights(
        origin_iata="aus",
        destination_iata="lis",
        departure_date="2026-10-10",
        return_date="2026-10-14",
        max_price_per_person_usd=800.0,
        travelers=2,
    )
    assert ok_res["status"] == "ok"
    assert len(ok_res["flights"]) == 2
    assert ok_res["flights"][0]["flight_id"] == "FLT-AUS-LIS-101"
    assert ok_res["lowest_total_usd"] == 1240.0

    bad_iata = toolset.search_budget_flights(
        origin_iata="AUSTIN",
        destination_iata="LIS",
        departure_date="2026-10-10",
        return_date="2026-10-14",
        max_price_per_person_usd=800.0,
    )
    assert bad_iata["status"] == "error"
    assert bad_iata["error_code"] == "INVALID_FLIGHT_SEARCH_ARGS"
    assert "recovery_instructions" in bad_iata
    assert bad_iata["suggested_next_tool"] == "search_budget_flights"

    over_budget = toolset.search_budget_flights(
        origin_iata="AUS",
        destination_iata="NRT",
        departure_date="2026-10-10",
        return_date="2026-10-15",
        max_price_per_person_usd=300.0,
    )
    assert over_budget["status"] == "error"
    assert over_budget["error_code"] == "NO_FLIGHTS_WITHIN_BUDGET"
    assert over_budget["details"]["lowest_available_per_person_usd"] == 780.0


def test_hotel_search_ranks_amenities_and_recovers_gracefully(
    toolset: TravelConciergeToolset,
) -> None:
    """Hotel search prioritizes requested amenities and computes tax-inclusive stay totals."""
    res = toolset.search_hotels_within_budget(
        destination_city="Tokyo",
        check_in_date="2026-11-01",
        check_out_date="2026-11-05",
        max_nightly_rate_usd=210.0,
        preferred_amenities=["transit", "quiet"],
    )
    assert res["status"] == "ok"
    assert res["nights"] == 4
    assert res["hotels"][0]["hotel_id"] == "HTL-NRT-002"
    assert res["hotels"][0]["total_stay_usd"] == pytest.approx(858.0, abs=0.01)

    unknown_city = toolset.search_hotels_within_budget(
        destination_city="Atlantis",
        check_in_date="2026-11-01",
        check_out_date="2026-11-05",
        max_nightly_rate_usd=200.0,
    )
    assert unknown_city["status"] == "error"
    assert unknown_city["error_code"] == "CITY_NOT_IN_CATALOG"
    assert "Tokyo" in unknown_city["recovery_instructions"]


def test_budget_breakdown_deterministic_compliance_and_overage_guidance(
    toolset: TravelConciergeToolset,
) -> None:
    """Budget breakdown flags over-budget plans with concrete savings levers."""
    within = toolset.calculate_trip_budget_breakdown(
        total_budget_usd=2000.0,
        flight_total_usd=620.0,
        hotel_total_usd=483.84,
        nights=4,
        travelers=1,
    )
    assert within["status"] == "ok"
    assert within["within_budget"] is True
    assert within["remaining_budget_usd"] > 0

    over = toolset.calculate_trip_budget_breakdown(
        total_budget_usd=1200.0,
        flight_total_usd=780.0,
        hotel_total_usd=600.0,
        nights=4,
        travelers=1,
    )
    assert over["status"] == "ok"
    assert over["within_budget"] is False
    assert over["over_budget_by_usd"] > 0
    assert any("exceeds hard ceiling" in rec for rec in over["savings_recommendations"])
