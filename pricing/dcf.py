import numpy as np

from pricing.base import Pricer
from instruments.base import Instrument

from instruments.bond import (
    ZeroCouponBond,
    CouponBond
)

class DCF(Pricer):
    SUPPORTED = (ZeroCouponBond, CouponBond)

    def __init__(self, instrument: Instrument, r: float) -> None:
        if not isinstance(instrument, self.SUPPORTED):
            raise TypeError(f"DCF cannot price {type(instrument).__name__}")
        super().__init__(instrument)
        self.i = instrument
        self.r = r  # risk free rate

    def price(self) -> float:
        if isinstance(self.i, ZeroCouponBond):
            return self._price_zcb()
        return self._price_cb()
    
    def _price_zcb(self) -> float:
        return self.i.fv * np.exp(-self.r * self.i.mat)
    
    def _price_cb(self) -> float:
        pv_coupons = sum(self.i.coupon * np.exp(-self.r * t) for t in self.i.cf_times)
        pv_face = self.i.fv * np.exp(-self.r * self.i.mat)

        return pv_coupons + pv_face
    
    def duration(self) -> float:
        if isinstance(self.i, ZeroCouponBond):
            return self.i.mat
        
        w = sum(t * self.i.coupon * np.exp(-self.r * t) for t in times)
        w += self.i.mat * self.i.fv * np.exp(-self.r * self.i.mat)

        return w / self.price()
    
    def dv0x(self, bp_shift: float = 0.0001) -> float:
        # assume that if the input rate is too high then we divide
        if bp_shift >= 0.01:    bp_shift = bp_shift / 100

        up = DCF(self.i, self.r + bp_shift).price()
        down = DCF(self.i, self.r - bp_shift).price()

        return (down - up) / 2

        
