from datetime import date
from dataclasses import dataclass

from curves import YieldCurve
from surfaces import VolSurface
from utils.day_count import DayCount

@dataclass(frozen=True)
class MarketContext:
    """
    MarketContext represents the state of the market at a given valuation date.

    This is the key class that updates pricing and instruments.
    It can be instanciated with the current market data, again and again via the `bump` method, and then used to price instruments.
    """
    valuation_date: date
    yield_curve: YieldCurve
    vol_surface: VolSurface
    spot_prices: dict[str, float]
    div_yields: dict[str, float] = dict[str, float]()
    
    day_count: DayCount = DayCount.ACT_365
    

    def bump(self, **kwargs) -> "MarketContext":
        """
        Return new MarketContext with selected fields updated
        """

        return MarketContext(
            valuation_date=kwargs.get("valuation_date", self.valuation_date),
            yield_curve=kwargs.get("yield_curve", self.yield_curve),
            vol_surface=kwargs.get("vol_surface", self.vol_surface),
            spot_prices=kwargs.get("spot_prices", self.spot_prices),
            div_yields=kwargs.get("div_yields", self.div_yields),
            day_count=kwargs.get("day_count", self.day_count)
        )
