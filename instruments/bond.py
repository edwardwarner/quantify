from dataclasses import dataclass, field
from instruments.base import Instrument

from typing import List

@dataclass
class ZeroCouponBond(Instrument):
    fv: float
    mat: float

@dataclass
class CouponBond(Instrument):
    fv: float
    mat: float
    coupon_rate: float
    freq: int

    coupon: float = field(init=False)
    adj_rate: float = field(init=False)
    periods: int = field(init=False)
    cf_times: List[float] = field(init=False)

    def __post_init__(self):
        self.adj_rate = self.coupon_rate / self.freq
        self.coupon = self.fv * self.adj_rate

        self.periods = int(self.mat * self.freq)
        self.cf_times = [t / self.freq for t in range(1, self.periods + 1)]
