import numpy as np
from typing import Any, Literal, Tuple

from .motion import Motion

class GeometricBrownianMotion(Motion):
    def __init__(self,
                 S0: float, mu: float, sigma: float,
                 T: float,
                 n_steps: int,
                 RANDOM_STATE: int = 7,
                 **kwargs: Any) -> None:
        super().__init__(T=T, n_steps=n_steps, RANDOM_STATE=RANDOM_STATE, **kwargs)

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def simulate(self, n_paths: int,
                 method: Literal["exact", "euler"] = "exact") -> np.ndarray:
        func = getattr(GeometricBrownianMotion, f"_simulate_{method}", None)
        if func is None:
            raise AttributeError(f"Unknown method: _simulate_{method}")
        return func(self, n_paths=n_paths)

    def _simulate_exact(self, n_paths: int) -> np.ndarray:
        """
        Simulating St = S0 * exp((mu - 1/2*sigma^2)*t + sigma*Wt)
        """
        dW = self.rng.normal(0, np.sqrt(self.dt), size = (n_paths, self.n_steps))
        W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis = 1)], axis = 1)
        drift = (self.mu - 0.5 * self.sigma**2) * self.t

        return self.S0 * np.exp(drift + self.sigma * W)
    
    def _simulate_euler(self,  n_paths: int) -> np.ndarray:
        pass
    
    def log_returns(self, S: np.ndarray):
        return np.log(S[:, -1] / self.S0)
    
    def theoretical_log_returns_mean_std(self) -> Tuple[float, float]:
        mean = (self.mu - 0.5 * self.sigma**2) * self.T
        std = self.sigma * np.sqrt(self.T)
        return (mean, std)
    
    def plot_paths_2d(self, S: np.ndarray, n_sims: int = 20) -> None:
        return super().plot_paths_2d(S, n_sims)

    def plot_final_values(self, S: np.ndarray) -> None:
        return super().plot_final_values(S)
    
    def plot_log_returns(self, lr: np.ndarray, th_mean: float, th_std: float) -> None:
        return super().plot_log_returns(lr, th_mean, th_std)
