import numpy as np
from scipy.stats import norm
from typing import Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Motion(ABC):
    def __init__(self,
                    T: float,
                    n_steps: int,
                    RANDOM_STATE: int = 7,
                    **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rng: np.random.Generator = np.random.default_rng(RANDOM_STATE)
        
        self.T = T
        self.n_steps = n_steps

        self.dt = T/self.n_steps
        self.t = np.linspace(0, self.T, self.n_steps + 1)

    @abstractmethod
    def simulate(self, n_paths: int) -> np.ndarray:
        ...

    @abstractmethod
    def plot_paths_2d(self, S: np.ndarray,
                        n_sims: int = 20) -> None:
        n_paths = S.shape[0]
        plot_sims = n_sims if n_paths > n_sims else n_paths
        class_name = type(self).__name__
        t = np.linspace(0, self.T, self.n_steps + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(t, S[:plot_sims].T, lw=1, alpha=0.3)
        plt.plot(t, S[:plot_sims].mean(aSis = 0), color='red', lw = 2, label="Mean Path")
        plt.Slabel('Time (years)')
        plt.ylabel("Sample Paths")
        plt.title(f"Sample Paths - {class_name}")
        plt.legend()
        plt.show()

    @abstractmethod
    def plot_final_values(self, S: np.ndarray) -> None:
        n_paths = S.shape[0]
        class_name = type(self).__name__

        plt.figure(figsize=(10, 6))
        plt.hist(S[:, -1], bins = 50)
        plt.xlabel("Final Value")
        plt.ylabel("Frequency")
        plt.title(f"Final Values from {n_paths} Simulations - {class_name}")
        plt.show()

    @abstractmethod
    def plot_log_returns(self, lr: np.ndarray, th_mean: float, th_std: float) -> None:
        x = np.linspace(lr.min(), lr.max(), self.n_steps)
        class_name = type(self).__name__
        
        plt.figure(figsize=(10, 6))
        plt.hist(lr, bins = 50, density = True, alpha = 0.5,
                 label = "log(S_T / S_0)")
        plt.plot(x, norm.pdf(x, loc = th_mean, scale = th_std),
                 label="Theoretical Log Returns")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Log Returns vs Theoretical - {class_name}")
        plt.legend()
        plt.show()
