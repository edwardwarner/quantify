import numpy as np
from typing import Any

from .motion import Motion

class BrownianMotion(Motion):
    def __init__(self,
                 T: int,
                 n_steps: int,
                 RANDOM_STATE: int = 7,
                 **kwargs: Any) -> None:
        super().__init__(T=T, n_steps=n_steps, RANDOM_STATE=RANDOM_STATE, **kwargs)
        self.mu = 0
        self.sigma = 1

    def simulate(self, n_paths:int) -> np.ndarray:
        X = np.zeros((n_paths, self.n_steps + 1))

        dX = self.rng.normal(loc = self.mu,
                                  scale = self.sigma * np.sqrt(self.dt),
                                  size = (n_paths, self.n_steps))
        X[:, 1:] = np.cumsum(dX, axis = 1)
        return X
    
    def plot_paths_2d(self, S: np.ndarray, n_sims: int = 20) -> None:
        return super().plot_paths_2d(S, n_sims)

    def plot_final_values(self, S: np.ndarray) -> None:
        return super().plot_final_values(S)
    
    def plot_log_returns(self, lr: np.ndarray[tuple[Any, ...], np.dtype[Any]], th_mean: float, th_std: float) -> None:
        return None
