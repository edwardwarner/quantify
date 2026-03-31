import pytest
import numpy as np

from dataclasses import dataclass

from stochastic import BrownianMotion, GeometricBrownianMotion

@dataclass
class Inputs:
    T: int = 1
    n_steps: int = 256
    cutoff: float = 0.05

    S0: float = 101.7
    mu: float = 0.06
    sigma: float = 0.19
    
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

class TestGeometricBrownianMotion:
    def setup_method(self):
        self.inputs = Inputs()
        self.gbm = GeometricBrownianMotion(self.inputs.S0,
                                            self.inputs.mu,
                                            self.inputs.sigma,
                                            self.inputs.T,
                                            self.inputs.n_steps)

    def test_output_shape(self):
        n_paths = self.inputs.n_paths()
        S = self.gbm.simulate(n_paths)
        assert S.shape == (n_paths, self.inputs.n_steps + 1), f"Output shape must be ({n_paths}, {self.inputs.n_steps + 1})"

    def test_S0(self):
        S = self.gbm.simulate(self.inputs.n_paths())
        assert np.allclose(S[:, 0], self.inputs.S0), "S_0 must equal initial price for all paths"

    def test_strictly_positive(self):
        S = self.gbm.simulate(self.inputs.n_paths())
        assert np.all(S > 0), "GBM paths must be strictly positive"

    def test_expected_value(self):
        """E[S_T] = S0 * exp(mu * T)"""
        S = self.gbm.simulate(self.inputs.n_paths())
        theoretical_mean = self.inputs.S0 * np.exp(self.inputs.mu * self.inputs.T)
        sample_mean = S[:, -1].mean()
        rel_error = abs(sample_mean - theoretical_mean) / theoretical_mean
        assert rel_error < self.inputs.cutoff, f"E[S_T] should be ~{theoretical_mean:.2f}, got {sample_mean:.2f}"

    def test_log_returns_mean(self):
        """E[log(S_T/S_0)] = (mu - 0.5*sigma^2) * T"""
        S = self.gbm.simulate(self.inputs.n_paths())
        lr = self.gbm.log_returns(S)
        theoretical_mean = (self.inputs.mu - 0.5 * self.inputs.sigma**2) * self.inputs.T
        sample_mean = lr.mean()
        assert abs(sample_mean - theoretical_mean) < self.inputs.cutoff, \
            f"E[log(S_T/S_0)] should be ~{theoretical_mean:.4f}, got {sample_mean:.4f}"

    def test_log_returns_std(self):
        """Std[log(S_T/S_0)] = sigma * sqrt(T)"""
        S = self.gbm.simulate(self.inputs.n_paths())
        lr = self.gbm.log_returns(S)
        theoretical_std = self.inputs.sigma * np.sqrt(self.inputs.T)
        sample_std = lr.std()
        assert abs(sample_std - theoretical_std) < self.inputs.cutoff, \
            f"Std[log(S_T/S_0)] should be ~{theoretical_std:.4f}, got {sample_std:.4f}"

    def test_log_returns_are_normal(self):
        """Log returns of GBM should be normally distributed"""
        from scipy.stats import kstest
        S = self.gbm.simulate(self.inputs.n_paths())
        lr = self.gbm.log_returns(S)
        th_mean, th_std = self.gbm.theoretical_log_returns_mean_std()
        _, p_value = kstest(lr, "norm", args=(th_mean, th_std))
        assert p_value > 0.01, f"Log returns failed normality test (p={p_value:.4f})"

    def test_final_values_are_lognormal(self):
        """S_T should be lognormally distributed"""
        from scipy.stats import kstest, lognorm
        S = self.gbm.simulate(self.inputs.n_paths())
        th_mean, th_std = self.gbm.theoretical_log_returns_mean_std()
        scale = self.inputs.S0 * np.exp(th_mean)
        _, p_value = kstest(S[:, -1], lognorm.cdf, args=(th_std, 0, scale))
        assert p_value > 0.01, f"Final values failed lognormality test (p={p_value:.4f})"

    def test_invalid_method_raises(self):
        with pytest.raises(AttributeError, match="Unknown method"):
            self.gbm.simulate(100, method="invalid")

    def test_reproducibility(self):
        gbm1 = GeometricBrownianMotion(self.inputs.S0, self.inputs.mu, self.inputs.sigma,
                                        self.inputs.T, self.inputs.n_steps, RANDOM_STATE=10)
        gbm2 = GeometricBrownianMotion(self.inputs.S0, self.inputs.mu, self.inputs.sigma,
                                        self.inputs.T, self.inputs.n_steps, RANDOM_STATE=10)
        np.testing.assert_array_equal(gbm1.simulate(100), gbm2.simulate(100))
