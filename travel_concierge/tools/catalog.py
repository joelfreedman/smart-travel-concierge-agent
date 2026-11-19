"""Deterministic synthetic flight and hotel inventory for reproducible planning and testing.

All prices and schedules are synthetic public sample data with zero external
API keys required for offline evaluation, unit tests, and local demos.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogFlight:
    """Immutable synthetic round-trip flight record."""

    flight_id: str
    airline: str
    origin_iata: str
    destination_iata: str
    cabin_class: str
    stops: int
    duration_hours: float
    price_per_person_usd: float
    refundable: bool


@dataclass(frozen=True)
class CatalogHotel:
    """Immutable synthetic hotel record."""

    hotel_id: str
    name: str
    city: str
    iata_code: str
    neighborhood: str
    nightly_rate_usd: float
    tax_rate: float
    rating: float
    amenities: tuple[str, ...]
    free_cancellation: bool


SUPPORTED_AIRPORTS: dict[str, str] = {
    "AUS": "Austin",
    "JFK": "New York",
    "SFO": "San Francisco",
    "ORD": "Chicago",
    "DEN": "Denver",
    "NRT": "Tokyo",
    "LIS": "Lisbon",
    "CDG": "Paris",
    "MEX": "Mexico City",
    "YVR": "Vancouver",
}

FLIGHT_CATALOG: tuple[CatalogFlight, ...] = (
    # Austin -> Tokyo
    CatalogFlight("FLT-AUS-NRT-101", "Pacific Air", "AUS", "NRT", "economy", 1, 15.5, 780.0, True),
    CatalogFlight("FLT-AUS-NRT-102", "SkyBridge", "AUS", "NRT", "economy", 0, 13.4, 960.0, True),
    CatalogFlight(
        "FLT-AUS-NRT-201", "Pacific Air", "AUS", "NRT", "premium_economy", 1, 15.0, 1390.0, True
    ),
    # Austin -> Lisbon
    CatalogFlight("FLT-AUS-LIS-101", "Atlantic Value", "AUS", "LIS", "economy", 1, 11.8, 620.0, True),
    CatalogFlight("FLT-AUS-LIS-102", "Iberia Express Demo", "AUS", "LIS", "economy", 1, 10.6, 740.0, False),
    CatalogFlight(
        "FLT-AUS-LIS-201", "Atlantic Value", "AUS", "LIS", "premium_economy", 1, 11.2, 1120.0, True
    ),
    # Austin -> Denver
    CatalogFlight("FLT-AUS-DEN-101", "FrontRange Jet", "AUS", "DEN", "economy", 0, 2.4, 195.0, True),
    CatalogFlight("FLT-AUS-DEN-102", "Summit Air", "AUS", "DEN", "economy", 0, 2.3, 245.0, True),
    CatalogFlight("FLT-AUS-DEN-201", "Summit Air", "AUS", "DEN", "business", 0, 2.3, 520.0, True),
    # New York -> Lisbon
    CatalogFlight("FLT-JFK-LIS-101", "Luzia Air", "JFK", "LIS", "economy", 0, 7.0, 540.0, True),
    CatalogFlight("FLT-JFK-LIS-102", "Atlantic Value", "JFK", "LIS", "economy", 1, 9.2, 460.0, False),
    CatalogFlight("FLT-JFK-LIS-201", "Luzia Air", "JFK", "LIS", "business", 0, 7.0, 1850.0, True),
    # San Francisco -> Tokyo
    CatalogFlight("FLT-SFO-NRT-101", "Pacific Air", "SFO", "NRT", "economy", 0, 10.9, 710.0, True),
    CatalogFlight("FLT-SFO-NRT-102", "Sunrise Jet", "SFO", "NRT", "economy", 1, 13.5, 595.0, False),
    CatalogFlight(
        "FLT-SFO-NRT-201", "Pacific Air", "SFO", "NRT", "premium_economy", 0, 10.9, 1280.0, True
    ),
    # Chicago -> Paris
    CatalogFlight("FLT-ORD-CDG-101", "EuroLink", "ORD", "CDG", "economy", 0, 8.4, 650.0, True),
    CatalogFlight("FLT-ORD-CDG-102", "BudgetWings", "ORD", "CDG", "economy", 1, 11.1, 520.0, False),
    # Austin -> Mexico City
    CatalogFlight("FLT-AUS-MEX-101", "Sol Airlines", "AUS", "MEX", "economy", 0, 2.5, 290.0, True),
    # San Francisco -> Vancouver
    CatalogFlight("FLT-SFO-YVR-101", "Cascadia Air", "SFO", "YVR", "economy", 0, 2.2, 230.0, True),
)

HOTEL_CATALOG: tuple[CatalogHotel, ...] = (
    # Tokyo
    CatalogHotel(
        "HTL-NRT-001",
        "Kanda Transit Inn",
        "Tokyo",
        "NRT",
        "Chiyoda",
        118.0,
        0.10,
        4.4,
        ("wifi", "transit", "breakfast"),
        True,
    ),
    CatalogHotel(
        "HTL-NRT-002",
        "Shibuya Park Boutique",
        "Tokyo",
        "NRT",
        "Shibuya",
        195.0,
        0.10,
        4.7,
        ("wifi", "transit", "gym", "quiet"),
        True,
    ),
    CatalogHotel(
        "HTL-NRT-003",
        "Ginza Grand Tower",
        "Tokyo",
        "NRT",
        "Ginza",
        340.0,
        0.10,
        4.9,
        ("wifi", "gym", "spa", "concierge"),
        False,
    ),
    # Lisbon
    CatalogHotel(
        "HTL-LIS-001",
        "Baixa Courtyard Hotel",
        "Lisbon",
        "LIS",
        "Baixa",
        112.0,
        0.08,
        4.5,
        ("wifi", "transit", "breakfast"),
        True,
    ),
    CatalogHotel(
        "HTL-LIS-002",
        "Chiado Terrace Suites",
        "Lisbon",
        "LIS",
        "Chiado",
        175.0,
        0.08,
        4.8,
        ("wifi", "transit", "quiet", "breakfast"),
        True,
    ),
    CatalogHotel(
        "HTL-LIS-003",
        "Avenida Palace Demo",
        "Lisbon",
        "LIS",
        "Avenida da Liberdade",
        310.0,
        0.08,
        4.9,
        ("wifi", "gym", "spa"),
        True,
    ),
    # Denver
    CatalogHotel(
        "HTL-DEN-001",
        "Union Station Rail Lodge",
        "Denver",
        "DEN",
        "LoDo",
        125.0,
        0.15,
        4.4,
        ("wifi", "transit", "breakfast"),
        True,
    ),
    CatalogHotel(
        "HTL-DEN-002",
        "Cherry Creek Alpine Stay",
        "Denver",
        "DEN",
        "Cherry Creek",
        185.0,
        0.15,
        4.7,
        ("wifi", "gym", "quiet"),
        True,
    ),
    # Paris
    CatalogHotel(
        "HTL-CDG-001",
        "Bastille Metro Hotel",
        "Paris",
        "CDG",
        "11th Arrondissement",
        142.0,
        0.10,
        4.4,
        ("wifi", "transit", "breakfast"),
        True,
    ),
    CatalogHotel(
        "HTL-CDG-002",
        "Marais Quiet Residence",
        "Paris",
        "CDG",
        "Le Marais",
        225.0,
        0.10,
        4.8,
        ("wifi", "transit", "quiet"),
        True,
    ),
    # Mexico City
    CatalogHotel(
        "HTL-MEX-001",
        "Roma Norte Casa",
        "Mexico City",
        "MEX",
        "Roma Norte",
        98.0,
        0.16,
        4.6,
        ("wifi", "breakfast", "quiet"),
        True,
    ),
    # Vancouver
    CatalogHotel(
        "HTL-YVR-001",
        "Yaletown Harbor Inn",
        "Vancouver",
        "YVR",
        "Yaletown",
        148.0,
        0.15,
        4.6,
        ("wifi", "transit", "gym"),
        True,
    ),
)


def lookup_flight_by_id(flight_id: str) -> CatalogFlight | None:
    """Return a catalog flight by exact ID, or None if not found."""
    for flight in FLIGHT_CATALOG:
        if flight.flight_id == flight_id:
            return flight
    return None


def lookup_hotel_by_id(hotel_id: str) -> CatalogHotel | None:
    """Return a catalog hotel by exact ID, or None if not found."""
    for hotel in HOTEL_CATALOG:
        if hotel.hotel_id == hotel_id:
            return hotel
    return None
