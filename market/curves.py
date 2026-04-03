import numpy as np
from datetime import date
from dataclasses import dataclass
from scipy.interpolate import CubicSpline, CubicHermiteSpline # type: ignore

from typing import Callable, Union, Protocol, runtime_checkable

from utils import BaseEnum, DayCount, day_count, year_fraction

class Interpolation(str, BaseEnum):
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    CUBIC_SPLINE = "cubic_spline"

@runtime_checkable
class YieldCurve(Protocol):

    def zero_rate(self, mat: date) -> float: ...
    def discount_factor(self, mat: date) -> float: ...
    def forward_rate(self, start:  date, end: date) -> float: ...


@dataclass(frozen=True)
class FlatCurve:
    """
    Flat yield curve for testing & BS.
    """
    valuation_date: date
    rate: float

    def zero_rate(self, mat: date) -> float:
        return self.rate
    
    def discount_factor(self, mat: date) -> float:
        return np.exp(-self.rate * year_fraction(self.valuation_date, mat, DayCount.ACT_365))
    
    def forward_rate(self, start:  date, end: date) -> float:
        return self.rate
    
    #TODO: par rate


# TODO we call day_count a lot here, maybe theres a way we can lessen that
# TODO freeze?
@dataclass
class BootstrappedCurve:
    valuation_date: date
    tenors: Union[tuple[date, ...], list[date]]
    rates: Union[tuple[float, ...], list[float], np.ndarray]
    
    day_count: DayCount = DayCount.ACT_365
    interpolation: Interpolation = Interpolation.CUBIC_SPLINE

    def __post_init__(self):
        self._tenors_yf = np.array([year_fraction(self.valuation_date, d, self.day_count) for d in self.tenors])
        self._interpolator = self._build_interpolator()

    def _build_interpolator(self) -> Callable[[float], float]:
        match self.interpolation:
            case Interpolation.LINEAR:
                return lambda t: float(np.interp(t, self._tenors_yf, self.rates))
            case Interpolation.LOG_LINEAR:
                log_dfs = [-r * tenor for r, tenor in zip(self.rates, self._tenors_yf)]
                return lambda t: -float(np.interp(t, self._tenors_yf, log_dfs)) / t
            case Interpolation.CUBIC_SPLINE:
                cs = CubicSpline(self._tenors_yf, self.rates)
                return lambda t: float(cs(t))

    def zero_rate(self, mat: date) -> float:
        t = day_count(mat, self.valuation_date, self.day_count)
        return self._interpolator(t)
    
    def discount_factor(self, mat: date) -> float:
        t = day_count(mat, self.valuation_date, self.day_count)
        return np.exp(-self.zero_rate(mat) * t)
    
    def forward_rate(self, start: date, end: date):
        """
        Derived from continuously compounded rate:
            exp(r2*t2) = exp(r1*t2) * exp(r1,2 * (t2 - t1))
        """
        t1 = day_count(start, self.valuation_date, self.day_count)
        t2 = day_count(end, self.valuation_date, self.day_count)

        df_t1 = self.discount_factor(start)
        df_t2 = self.discount_factor(end)

        return -np.log((df_t1/df_t2)) * 1/(t2 - t1)

    # TODO: par rate?
