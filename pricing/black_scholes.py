import numpy as np
from scipy.stats import norm

class BlackScholes:
    """ European Option Pricing using closed-form BS """
    def __init__(self,
                 S: float,
                 K: float,
                 r: float,
                 vol: float,
                 T: float) -> None:
        self.S = S
        self.K = K
        self.r = r
        self.vol = vol
        self.T = T

    @property
    def d1(self) -> float:
        return (np.log(self.S/self.K) + (self.r + 0.5*self.vol**2)*self.T)/(self.vol*np.sqrt(self.T))
    
    @property
    def d2(self) -> float:
        return self.d1 - self.vol*np.sqrt(self.T)
    
    @property
    def call(self) -> float:
        return norm.cdf(self.d1)*self.S - norm.cdf(self.d2)*self.K*np.exp(-self.r*self.T)
    
    @property
    def put(self) -> float:
        return norm.cdf(-self.d2)*self.K*np.exp(-self.r*self.T) - norm.cdf(-self.d1)*self.S


class Greeks(BlackScholes):
    def __init__(self, 
                 S: float, 
                 K: float, 
                 r: float, 
                 vol: float, 
                 T: float,
                 opt_type: str = "call") -> None:
        super().__init__(S, K, r, vol, T)
        self.opt_type = opt_type
    
    @property
    def delta(self) -> float:
        return norm.cdf(self.d1) if self.opt_type == "call" else norm.cdf(self.d1) - 1

    @property
    def gamma(self) -> float:
        return norm.pdf(self.d1) / (self.S * self.vol * np.sqrt(self.T))
    
    @property
    def vega(self) -> float:
        return self.S*norm.pdf(self.d1)*np.sqrt(self.T)
    
    @property
    def theta(self) -> float:
        d2_cdf = norm.cdf(self.d2) if self.opt_type == "call" else norm.cdf(-self.d2)
        quotient = -((self.S*norm.pdf(self.d1)*self.vol)/(2*np.sqrt(self.T)))

        return quotient + self.r*self.K*np.exp(-self.r*self.T)*d2_cdf * (-1 if self.opt_type == "call" else 1)
    
    @property
    def rho(self) -> float:
        d2_cdf = norm.cdf(self.d2) if self.opt_type == "call" else norm.cdf(-self.d2)

        return self.K*self.T*np.exp(-self.r*self.T)*d2_cdf * (1 if self.opt_type == "call" else -1)

