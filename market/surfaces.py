from datetime import date

from typing import Protocol, runtime_checkable

@runtime_checkable
class VolSurface(Protocol):
    
    def implied_vol(self, strike: float, expiry: date) -> float: ...
    def local_vol(self, spot: float, t: float) -> float: ...
