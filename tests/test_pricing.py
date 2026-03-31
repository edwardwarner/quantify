import pytest
import numpy as np

from dataclasses import dataclass

from pricing import BlackScholes, Greeks, MonteCarlo

@dataclass
class Inputs:
    S: float = 100.7
    K: float = 120.3
    r: float = 0.05
    vol: float = 0.18
    T: float = 2.0
    opt_type: str = "call"

    n_steps: int = 252
    n_paths: int = 10_000


class TestBlackScholes:
    def setup_method(self):
        self.inputs = Inputs()
        self.bs = BlackScholes(self.inputs.S,
                               self.inputs.K,
                               self.inputs.r,
                               self.inputs.K,
                               self.inputs.T)

    def test_d1_d2_relationship(self):
        """d2 = d1 - vol * sqrt(T)"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        expected_diff = self.inputs.vol * np.sqrt(self.inputs.T)
        assert abs((bs.d1 - bs.d2) - expected_diff) < 1e-10, \
            f"d1 - d2 should equal vol*sqrt(T) = {expected_diff}"

    def test_d1_computation(self):
        """Verify d1 against manual calculation"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        S, K, r, vol, T = self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T
        expected_d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        assert abs(bs.d1 - expected_d1) < 1e-10

    def test_call_price_positive(self):
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs.call > 0, "Call price must be positive"

    def test_put_price_positive(self):
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs.put > 0, "Put price must be positive"

    def test_put_call_parity(self):
        """C - P = S - K * exp(-rT)"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        lhs = bs.call - bs.put
        rhs = self.inputs.S - self.inputs.K * np.exp(-self.inputs.r * self.inputs.T)
        assert abs(lhs - rhs) < 1e-10, \
            f"Put-call parity violated: C-P={lhs:.6f}, S-Ke^(-rT)={rhs:.6f}"

    def test_call_upper_bound(self):
        """Call price <= S"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs.call <= self.inputs.S, "Call price must not exceed spot price"

    def test_put_upper_bound(self):
        """Put price <= K * exp(-rT)"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs.put <= self.inputs.K * np.exp(-self.inputs.r * self.inputs.T), \
            "Put price must not exceed discounted strike"

    def test_deep_itm_call(self):
        """Deep ITM call should be close to S - K*exp(-rT)"""
        bs = BlackScholes(200, 50, self.inputs.r, self.inputs.vol, self.inputs.T)
        intrinsic = 200 - 50 * np.exp(-self.inputs.r * self.inputs.T)
        assert abs(bs.call - intrinsic) / intrinsic < 0.01

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero"""
        bs = BlackScholes(50, 200, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs.call < 1.0, f"Deep OTM call should be near zero, got {bs.call}"

    def test_call_increases_with_spot(self):
        """Call price should increase as spot increases"""
        bs_low = BlackScholes(90, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        bs_high = BlackScholes(110, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs_high.call > bs_low.call

    def test_put_increases_with_strike(self):
        """Put price should increase as strike increases"""
        bs_low = BlackScholes(self.inputs.S, 100, self.inputs.r, self.inputs.vol, self.inputs.T)
        bs_high = BlackScholes(self.inputs.S, 140, self.inputs.r, self.inputs.vol, self.inputs.T)
        assert bs_high.put > bs_low.put

    def test_price_increases_with_volatility(self):
        """Both call and put prices should increase with volatility"""
        bs_low = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, 0.10, self.inputs.T)
        bs_high = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, 0.30, self.inputs.T)
        assert bs_high.call > bs_low.call
        assert bs_high.put > bs_low.put


class TestGreeks:
    def setup_method(self):
        self.inputs = Inputs()
        self.call_greeks = Greeks(self.inputs.S,
                                  self.inputs.K,
                                  self.inputs.r,
                                  self.inputs.vol,
                                  self.inputs.T,
                                  "call")
        self.put_greeks = Greeks(self.inputs.S,
                                 self.inputs.K,
                                 self.inputs.r,
                                 self.inputs.vol,
                                 self.inputs.T,
                                 "put")

    def test_call_delta_bounds(self):
        """Call delta must be in [0, 1]"""
        assert 0 <= self.call_greeks.delta <= 1, \
            f"Call delta out of bounds: {self.call_greeks.delta}"

    def test_put_delta_bounds(self):
        """Put delta must be in [-1, 0]"""
        assert -1 <= self.put_greeks.delta <= 0, \
            f"Put delta out of bounds: {self.put_greeks.delta}"

    def test_put_call_delta_relationship(self):
        """delta_call - delta_put = 1"""
        diff = self.call_greeks.delta - self.put_greeks.delta
        assert abs(diff - 1.0) < 1e-10, \
            f"delta_call - delta_put should equal 1, got {diff}"

    def test_gamma_positive(self):
        assert self.call_greeks.gamma > 0, "Gamma must be positive"

    def test_gamma_same_for_call_and_put(self):
        """Gamma is the same for call and put"""
        assert abs(self.call_greeks.gamma - self.put_greeks.gamma) < 1e-10

    def test_vega_positive(self):
        assert self.call_greeks.vega > 0, "Vega must be positive"

    def test_vega_same_for_call_and_put(self):
        """Vega is the same for call and put"""
        assert abs(self.call_greeks.vega - self.put_greeks.vega) < 1e-10

    def test_call_rho_positive(self):
        """Call rho should be positive (call value increases with rates)"""
        assert self.call_greeks.rho > 0

    def test_put_rho_negative(self):
        """Put rho should be negative (put value decreases with rates)"""
        assert self.put_greeks.rho < 0

    def test_delta_finite_difference(self):
        """Verify analytical delta against finite difference approximation"""
        h = 0.01
        bs_up = BlackScholes(self.inputs.S + h, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        bs_dn = BlackScholes(self.inputs.S - h, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        fd_delta = (bs_up.call - bs_dn.call) / (2 * h)
        assert abs(self.call_greeks.delta - fd_delta) < 1e-4, \
            f"Analytical delta {self.call_greeks.delta:.6f} != FD delta {fd_delta:.6f}"

    def test_gamma_finite_difference(self):
        """Verify analytical gamma against finite difference approximation"""
        h = 0.01
        bs_up = BlackScholes(self.inputs.S + h, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        bs_mid = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        bs_dn = BlackScholes(self.inputs.S - h, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T)
        fd_gamma = (bs_up.call - 2 * bs_mid.call + bs_dn.call) / (h ** 2)
        assert abs(self.call_greeks.gamma - fd_gamma) < 1e-4, \
            f"Analytical gamma {self.call_greeks.gamma:.6f} != FD gamma {fd_gamma:.6f}"

    def test_vega_finite_difference(self):
        """Verify analytical vega against finite difference approximation"""
        h = 0.0001
        bs_up = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol + h, self.inputs.T)
        bs_dn = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol - h, self.inputs.T)
        fd_vega = (bs_up.call - bs_dn.call) / (2 * h)
        assert abs(self.call_greeks.vega - fd_vega) < 1e-2, \
            f"Analytical vega {self.call_greeks.vega:.6f} != FD vega {fd_vega:.6f}"

    def test_theta_finite_difference(self):
        """Verify analytical theta against finite difference approximation: theta = -dC/dT"""
        h = 1/365
        bs_up = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T + h)
        bs_dn = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol, self.inputs.T - h)
        fd_theta = -(bs_up.call - bs_dn.call) / (2 * h)
        assert abs(self.call_greeks.theta - fd_theta) < 0.1, \
            f"Analytical theta {self.call_greeks.theta:.6f} != FD theta {fd_theta:.6f}"

    def test_rho_finite_difference(self):
        """Verify analytical rho against finite difference approximation"""
        h = 0.0001
        bs_up = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r + h, self.inputs.vol, self.inputs.T)
        bs_dn = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r - h, self.inputs.vol, self.inputs.T)
        fd_rho = (bs_up.call - bs_dn.call) / (2 * h)
        assert abs(self.call_greeks.rho - fd_rho) < 0.1, \
            f"Analytical rho {self.call_greeks.rho:.6f} != FD rho {fd_rho:.6f}"

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be close to 0.5 (slightly above due to drift)"""
        atm = Greeks(100, 100, self.inputs.r, self.inputs.vol, self.inputs.T, "call")
        assert 0.45 < atm.delta < 0.75, f"ATM call delta should be near 0.5, got {atm.delta}"

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call delta should approach 1"""
        deep_itm = Greeks(200, 50, self.inputs.r, self.inputs.vol, self.inputs.T, "call")
        assert deep_itm.delta > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        """Deep OTM call delta should approach 0"""
        deep_otm = Greeks(50, 200, self.inputs.r, self.inputs.vol, self.inputs.T, "call")
        assert deep_otm.delta < 0.01

class TestMonteCarlo:
    def setup_method(self):
        self.inputs = Inputs()
        np.random.seed(42)
        self.mc_call = MonteCarlo(self.inputs.S,
                                  self.inputs.K,
                                  self.inputs.r,
                                  self.inputs.vol,
                                  self.inputs.T,
                                  self.inputs.n_paths,
                                  self.inputs.n_steps,
                                  "call")
        self.mc_put = MonteCarlo(self.inputs.S,
                                 self.inputs.K,
                                 self.inputs.r,
                                 self.inputs.vol,
                                 self.inputs.T,
                                 self.inputs.n_paths,
                                 self.inputs.n_steps,
                                 "put")

    def test_call_price_positive(self):
        assert self.mc_call.price() > 0, "MC call price must be positive"

    def test_put_price_positive(self):
        assert self.mc_put.price() > 0, "MC put price must be positive"

    def test_call_upper_bound(self):
        """Call price <= S"""
        assert self.mc_call.price() <= self.inputs.S

    def test_put_upper_bound(self):
        """Put price <= K * exp(-rT)"""
        assert self.mc_put.price() <= self.inputs.K * np.exp(-self.inputs.r * self.inputs.T)

    def test_deep_itm_call(self):
        """Deep ITM call should be close to S - K*exp(-rT)"""
        mc = MonteCarlo(200, 50, self.inputs.r, self.inputs.vol, self.inputs.T,
                        50_000, self.inputs.n_steps, "call")
        intrinsic = 200 - 50 * np.exp(-self.inputs.r * self.inputs.T)
        assert abs(mc.price() - intrinsic) / intrinsic < 0.02

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero"""
        mc = MonteCarlo(50, 200, self.inputs.r, self.inputs.vol, self.inputs.T,
                        self.inputs.n_paths, self.inputs.n_steps, "call")
        assert mc.price() < 1.0

    def test_call_increases_with_spot(self):
        """Call price should increase as spot increases"""
        mc_low = MonteCarlo(90, self.inputs.K, self.inputs.r, self.inputs.vol,
                            self.inputs.T, 50_000, self.inputs.n_steps, "call")
        mc_high = MonteCarlo(110, self.inputs.K, self.inputs.r, self.inputs.vol,
                             self.inputs.T, 50_000, self.inputs.n_steps, "call")
        assert mc_high.price() > mc_low.price()

    def test_put_increases_with_strike(self):
        """Put price should increase as strike increases"""
        mc_low = MonteCarlo(self.inputs.S, 100, self.inputs.r, self.inputs.vol,
                            self.inputs.T, 50_000, self.inputs.n_steps, "put")
        mc_high = MonteCarlo(self.inputs.S, 140, self.inputs.r, self.inputs.vol,
                             self.inputs.T, 50_000, self.inputs.n_steps, "put")
        assert mc_high.price() > mc_low.price()

    def test_price_increases_with_volatility(self):
        """Both call and put prices should increase with volatility"""
        mc_low = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, 0.10,
                            self.inputs.T, 50_000, self.inputs.n_steps, "call")
        mc_high = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, 0.30,
                             self.inputs.T, 50_000, self.inputs.n_steps, "call")
        assert mc_high.price() > mc_low.price()

    def test_put_call_parity(self):
        """MC call - MC put should approximate S - K*exp(-rT)"""
        n = 100_000
        mc_c = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol,
                          self.inputs.T, n, self.inputs.n_steps, "call")
        mc_p = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol,
                          self.inputs.T, n, self.inputs.n_steps, "put")
        lhs = mc_c.price() - mc_p.price()
        rhs = self.inputs.S - self.inputs.K * np.exp(-self.inputs.r * self.inputs.T)
        assert abs(lhs - rhs) < 1.0, \
            f"Put-call parity violated: C-P={lhs:.4f}, S-Ke^(-rT)={rhs:.4f}"

    # --- Monte Carlo vs Black-Scholes convergence tests ---

    def test_mc_call_converges_to_bs(self):
        """MC call price should converge to BS analytical call price"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r,
                          self.inputs.vol, self.inputs.T)
        mc = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol,
                        self.inputs.T, 100_000, self.inputs.n_steps, "call")
        mc_price = mc.price()
        assert abs(mc_price - bs.call) / bs.call < 0.05, \
            f"MC call {mc_price:.4f} too far from BS call {bs.call:.4f}"

    def test_mc_put_converges_to_bs(self):
        """MC put price should converge to BS analytical put price"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r,
                          self.inputs.vol, self.inputs.T)
        mc = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol,
                        self.inputs.T, 100_000, self.inputs.n_steps, "put")
        mc_price = mc.price()
        assert abs(mc_price - bs.put) / bs.put < 0.05, \
            f"MC put {mc_price:.4f} too far from BS put {bs.put:.4f}"

    def test_mc_atm_call_converges_to_bs(self):
        """ATM MC call should match BS closely"""
        S = K = 100.0
        bs = BlackScholes(S, K, self.inputs.r, self.inputs.vol, self.inputs.T)
        mc = MonteCarlo(S, K, self.inputs.r, self.inputs.vol, self.inputs.T,
                        100_000, self.inputs.n_steps, "call")
        mc_price = mc.price()
        assert abs(mc_price - bs.call) / bs.call < 0.05, \
            f"ATM MC call {mc_price:.4f} too far from BS call {bs.call:.4f}"

    def test_mc_itm_put_converges_to_bs(self):
        """ITM put (S < K) MC should match BS"""
        S, K = 90.0, 110.0
        bs = BlackScholes(S, K, self.inputs.r, self.inputs.vol, self.inputs.T)
        mc = MonteCarlo(S, K, self.inputs.r, self.inputs.vol, self.inputs.T,
                        100_000, self.inputs.n_steps, "put")
        mc_price = mc.price()
        assert abs(mc_price - bs.put) / bs.put < 0.05, \
            f"ITM MC put {mc_price:.4f} too far from BS put {bs.put:.4f}"

    def test_mc_accuracy_improves_with_paths(self):
        """More paths should yield a price closer to BS"""
        bs = BlackScholes(self.inputs.S, self.inputs.K, self.inputs.r,
                          self.inputs.vol, self.inputs.T)
        mc_few = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol,
                            self.inputs.T, 1_000, self.inputs.n_steps, "call")
        mc_many = MonteCarlo(self.inputs.S, self.inputs.K, self.inputs.r, self.inputs.vol,
                             self.inputs.T, 200_000, self.inputs.n_steps, "call")
        err_few = abs(mc_few.price() - bs.call)
        err_many = abs(mc_many.price() - bs.call)
        assert err_many < err_few, \
            f"More paths should reduce error: err(1k)={err_few:.4f}, err(200k)={err_many:.4f}"
