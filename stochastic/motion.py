import numpy as np
from typing import Any
from abc import ABC, abstractmethod

class Motion(ABC):
    def __init__(self,
                    T: int,
                    n_steps: int,
                    RANDOM_STATE: int = 7,
                    **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rng: np.random.Generator = np.random.default_rng(RANDOM_STATE)
        
        self.T = T
        self.n_steps = n_steps

        self.dt = T/self.n_steps

    @abstractmethod
    def simulate(self, n_paths: int) -> np.ndarray:
        ...
