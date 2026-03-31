import numpy as np
from scipy.stats import norm

from pricing.base import Pricer
from instruments.base import Instrument

from instruments.option import (
    EuropeanOption
)

class BlackScholes(Pricer):
    SUPPORTED = (EuropeanOption)

    def __init__(self, instrument: Instrument) -> None:
        if not isinstance(instrument, self.SUPPORTED):
            raise TypeError(f"BlackScholes cannot price {type(instrument).__name__}")
        super().__init__(instrument)
        self.i = instrument

    @property
    def d1(self) -> float:
        return (np.log(self.i.S/self.i.K) + (self.i.r + 0.5*self.i.vol**2)*self.i.T)/(self.i.vol*np.sqrt(self.i.T))
    
    @property
    def d2(self) -> float:
        return self.d1 - self.i.vol*np.sqrt(self.i.T)
    
    def price(self) -> float:
        if self.i.opt_type == "call":
            return norm.cdf(self.d1)*self.i.S - norm.cdf(self.d2)*self.i.K*np.exp(-self.i.r*self.i.T)
        else:
            return norm.cdf(-self.d2)*self.i.K*np.exp(-self.i.r*self.i.T) - norm.cdf(-self.d1)*self.i.S
        
    # Greeks
    @property
    def delta(self) -> float:
        return norm.cdf(self.d1) if self.i.opt_type == "call" else norm.cdf(self.d1) - 1

    @property
    def gamma(self) -> float:
        return norm.pdf(self.d1) / (self.i.S * self.i.vol * np.sqrt(self.i.T))
    
    @property
    def vega(self) -> float:
        return self.i.S*norm.pdf(self.d1)*np.sqrt(self.i.T)
    
    @property
    def theta(self) -> float:
        d2_cdf = norm.cdf(self.d2) if self.i.opt_type == "call" else norm.cdf(-self.d2)
        quotient = -((self.i.S*norm.pdf(self.d1)*self.i.vol)/(2*np.sqrt(self.i.T)))

        return quotient + self.i.r*self.i.K*np.exp(-self.i.r*self.i.T)*d2_cdf * (-1 if self.i.opt_type == "call" else 1)
    
    @property
    def rho(self) -> float:
        d2_cdf = norm.cdf(self.d2) if self.i.opt_type == "call" else norm.cdf(-self.d2)

        return self.i.K*self.i.T*np.exp(-self.i.r*self.i.T)*d2_cdf * (1 if self.i.opt_type == "call" else -1)
    


