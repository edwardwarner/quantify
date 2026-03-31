import pytest
import numpy as np

from dataclasses import dataclass

from stochastic import BrownianMotion

@dataclass
class Inputs:
    T: int = 1
    n_steps: int = 256
    cutoff: float = 0.05
    
    def n_paths(self) -> int:
        return np.random.randint(10, 10_000)


class TestBrownianMotion:
    def setup_method(self):
        self.inputs = Inputs()
        self.bm = BrownianMotion(self.inputs.T, self.inputs.n_steps)

    def test_output_shape(self):
        n_paths = self.inputs.n_paths()
        X = self.bm.simulate(n_paths)
        assert X.shape == (n_paths, self.inputs.n_steps + 1), f"Output shape must be {self.inputs.n_steps}, ${self.inputs.n_paths}"
    
    def test_X_0(self):
        X = self.bm.simulate(self.inputs.n_paths())
        assert np.all(X[:, 0] == 0.0), "X_0 must be zero for all paths"

    def test_zero_mean(self):
        X = self.bm.simulate(self.inputs.n_paths())
        end_mean = X[:, -1].mean()
        assert abs(end_mean) < self.inputs.cutoff, f"E[X_T] should be ~0, got {end_mean:.4f}"

    def test_variance_equals_time(self):
        X = self.bm.simulate(self.inputs.n_paths())

        for moment in [0.25, 0.5, 0.75, 1]:
            var_at_moment = X[:, int(self.inputs.n_steps * moment)].var() if moment != 1 else X[:, -1].var()

            assert(var_at_moment - moment) < self.inputs.cutoff, f"Var[X_{moment}] should equal {moment}, got {var_at_moment: .4f}"

    def test_increments_are_normal(self):
        from scipy.stats import kstest, norm
        X = self.bm.simulate(self.inputs.n_paths())

        increments = X[:, -1] - X[:, 0]
        _, p_value = kstest(increments, "norm")
        assert p_value > 0.01, f"Increments failed normality test (p={p_value:.4f})"

    def test_reproducibility(self):
        bm1 = BrownianMotion(self.inputs.T, self.inputs.n_steps, RANDOM_STATE= 10)
        bm2 = BrownianMotion(self.inputs.T, self.inputs.n_steps, RANDOM_STATE= 10)

        np.testing.assert_array_equal(bm1.simulate(100), bm2.simulate(100))
