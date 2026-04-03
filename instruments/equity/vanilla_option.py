from datetime import date
from dataclasses import dataclass
from typing import Literal

from instruments.base import PayoffSmoothness
from ...utils.day_count import DayCount, year_fraction

@dataclass  #(frozen = True)
class VanillaOption:
    """
    Base class for vanilla options, not to be instantiated directly.
    Classes that inherit from this should implement the payoff method, to be BSPriceable.
    """
    underlying: str
    strike: float
    expiry: date
    currency: str

    day_count: DayCount = DayCount.ACT_365
    opt_type: Literal["call", "put"] = "call"

    def time_to_expiry(self, valuation_date: date) -> float:
        return year_fraction(valuation_date, self.expiry, self.day_count)
            
@dataclass  #(frozen = True)
class EuropeanOption(VanillaOption):
    instrument_id: str = "euro_opt"
    
    asset_class: str = "equity"
    payoff_smoothness: PayoffSmoothness = PayoffSmoothness.SMOOTH

    def payoff(self, S_T: float) -> float:
        match self.opt_type:
            case "call":
                return max(0.0, S_T - self.strike)
            case "put":
                return max(0.0, self.strike - S_T)

@dataclass  #(frozen = True)
class BinaryOption(VanillaOption):
    instrument_id: str = "bin_opt"
    
    cash_amnt: float = 1.0
    asset_class: str = "equity"
    payoff_smoothness: PayoffSmoothness = PayoffSmoothness.DISCONTINUOUS

    def payoff(self, S_T: float) -> float:
        match self.opt_type:
            case "call":
                return self.cash_amnt if S_T > self.strike else 0.0
            case "put":
                return self.cash_amnt if S_T < self.strike else 0.0
