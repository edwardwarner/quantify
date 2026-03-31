from typing import Any

import numpy as np

from pricing.base import Pricer
from instruments.base import Instrument

from instruments.option import (
    EuropeanOption
)
from stochastic.gbm import GeometricBrownianMotion

class MonteCarlo(Pricer):
    SUPPORTED = (EuropeanOption)

    def __init__(self, instrument: Instrument, n_paths: int = 10_000, n_steps: int = 252) -> None:
        if not isinstance(instrument, self.SUPPORTED):
            raise TypeError(f"MonteCarlo cannot price {type(instrument).__name__}")
        super().__init__(instrument)
        
        self.i = instrument
        self.n_paths = n_paths
        self.n_steps = n_steps

    def price(self) -> float:
        gbm = GeometricBrownianMotion(
            self.i.S, self.i.r, self.i.vol, self.i.T, self.n_steps
        )
        paths = gbm.simulate(self.n_paths)

        if self.i.opt_type == "call":
            payoffs = np.maximum(paths[:, -1] - self.i.K, 0)
        else:
            payoffs = np.maximum(self.i.K - paths[:, -1], 0)
        
        return float(np.exp(-self.i.r * self.i.T) * np.mean(payoffs))
