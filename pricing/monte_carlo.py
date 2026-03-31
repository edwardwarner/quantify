import numpy as np
from stochastic.gbm import GeometricBrownianMotion

class MonteCarlo:
    def __init__(self,
                 S: float,
                 K: float,
                 r: float,
                 vol: float,
                 T: float,
                 n_paths: int = 10_000,
                 n_steps: int = 252,
                 opt_type: str = "call") -> None:
        self.S = S
        self.K = K
        self.r = r
        self.vol = vol
        self.T = T

        self.n_paths = n_paths
        self.n_steps = n_steps

        self.opt_type = opt_type

        self.payoffs: float = 0.0

    def price(self) -> float:
        gbm = GeometricBrownianMotion(
            self.S, self.r, self.vol, self.T, self.n_steps
        )
        paths = gbm.simulate(self.n_paths)

        if self.opt_type == "call":
            payoffs = np.maximum(paths[:, -1] - self.K, 0)
        else:
            payoffs = np.maximum(self.K - paths[:, -1], 0)
        
        return float(np.exp(-self.r * self.T) * np.mean(payoffs))
