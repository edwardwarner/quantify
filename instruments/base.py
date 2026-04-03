from datetime import date
from enum import auto
from typing import Literal, Protocol, runtime_checkable, TypeVar

from utils import BaseEnum, DayCount

class PayoffSmoothness(BaseEnum):
    """Enum for payoff smoothness levels"""
    SMOOTH = auto()
    DISCONTINUOUS = auto()
    MIXED = auto()

InstrumentT = TypeVar("InstrumentT", bound="Instrument")

@runtime_checkable
class Instrument(Protocol):
    """
    Base protocol for all financial instruments.

    Day count field allows for instruments to specify their own day count convention.
    """

    instrument_id: str
    asset_class: str
    currency: str

    day_count: DayCount = DayCount.ACT_365

@runtime_checkable
class BSPriceable(Instrument, Protocol):
    """
    Protocol for instruments that can be priced using the Black-Scholes framework. 
    
    Requires payoff and time to expiry functions.
    """
    underlying: str
    strike: float
    expiry: date
    opt_type: Literal["call", "put"]

    def time_to_expiry(self, valuation_date: date) -> float: ...
    def payoff(self, S_T: float) -> float: ...
