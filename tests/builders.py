"""Small constructors for test data."""
from datetime import datetime

from travelex.schemas import (
    GroupResult, Leg, Money, RouteOption, Stop, StopCosts, Traveller, TripRequest, VisaCheck,
)
from travelex.tools import fees
from travelex.tools.visa_rules import lookup
from travelex.trip import group_by_origin

BHUTAN = Stop(place="Bhutan", country="BT")


def person(label: str, type: str = "adult", age: int | None = None, passport: str | None = "IN",
           origin: str = "Chennai", **kwargs) -> Traveller:
    return Traveller(label=label, type=type, age=age, passport=passport, origin=origin, **kwargs)


def trip(*travellers: Traveller, stops: list[Stop] = (BHUTAN,), nights: int | None = 4,
         budget: tuple[float, str] | None = (1500, "USD"), **kwargs) -> TripRequest:
    request = TripRequest(stops=list(stops), travellers=list(travellers), nights=nights,
                          budget=Money(amount=budget[0], currency=budget[1]) if budget else None, **kwargs)
    return request.model_copy(update={"groups": group_by_origin(request.travellers)})


def option(route: str, fare: float, currency: str = "USD", arrive: datetime | None = None,
           mode: str = "flight", **kwargs) -> RouteOption:
    stops = route.split("-")
    legs = [Leg(origin=a, destination=b, mode="flight") for a, b in zip(stops, stops[1:])]
    legs[-1] = legs[-1].model_copy(update={"arrive": arrive})
    return RouteOption(mode=mode, route=route, legs=legs, adult_fare=Money(amount=fare, currency=currency), **kwargs)


def result(group_id: str, *options: RouteOption, visas: list[VisaCheck] = ()) -> GroupResult:
    return GroupResult(group_id=group_id, options=list(options), visas=list(visas))


def visa(t: Traveller, country: str = "BT") -> VisaCheck:
    return VisaCheck(traveller=t.label, passport=t.passport, country=country, rule=lookup(t.passport, country))


def stop_costs(place: str = "Bhutan", currency: str = "USD", fee_country: str | None = "BT", **kwargs) -> StopCosts:
    values = dict(place=place, currency=currency, lodging_per_room_night=60, rooms=1, room_config="1 double",
                  food_per_person_day=15, fees=fees.rules_for(fee_country) if fee_country else [])
    return StopCosts(**{**values, **kwargs})
