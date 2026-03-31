import pytest
import numpy as np

from instruments.option import EuropeanOption
from instruments.bond import ZeroCouponBond, CouponBond
from pricing import BlackScholes, MonteCarlo, DCF


def make_option(**overrides):
    defaults = dict(S=100.7, K=120.3, r=0.05, vol=0.18, T=2.0, opt_type="call")
    defaults.update(overrides)
    return EuropeanOption(**defaults)


class TestBlackScholes:
    def test_d1_d2_relationship(self):
        """d2 = d1 - vol * sqrt(T)"""
        opt = make_option()
        bs = BlackScholes(opt)
        expected_diff = opt.vol * np.sqrt(opt.T)
        assert abs((bs.d1 - bs.d2) - expected_diff) < 1e-10

    def test_d1_computation(self):
        """Verify d1 against manual calculation"""
        opt = make_option()
        bs = BlackScholes(opt)
        expected_d1 = (np.log(opt.S/opt.K) + (opt.r + 0.5*opt.vol**2)*opt.T) / (opt.vol*np.sqrt(opt.T))
        assert abs(bs.d1 - expected_d1) < 1e-10

    def test_call_price_positive(self):
        bs = BlackScholes(make_option(opt_type="call"))
        assert bs.price() > 0, "Call price must be positive"

    def test_put_price_positive(self):
        bs = BlackScholes(make_option(opt_type="put"))
        assert bs.price() > 0, "Put price must be positive"

    def test_put_call_parity(self):
        """C - P = S - K * exp(-rT)"""
        opt_c = make_option(opt_type="call")
        opt_p = make_option(opt_type="put")
        call_price = BlackScholes(opt_c).price()
        put_price = BlackScholes(opt_p).price()
        lhs = call_price - put_price
        rhs = opt_c.S - opt_c.K * np.exp(-opt_c.r * opt_c.T)
        assert abs(lhs - rhs) < 1e-10, \
            f"Put-call parity violated: C-P={lhs:.6f}, S-Ke^(-rT)={rhs:.6f}"

    def test_call_upper_bound(self):
        """Call price <= S"""
        opt = make_option(opt_type="call")
        assert BlackScholes(opt).price() <= opt.S

    def test_put_upper_bound(self):
        """Put price <= K * exp(-rT)"""
        opt = make_option(opt_type="put")
        assert BlackScholes(opt).price() <= opt.K * np.exp(-opt.r * opt.T)

    def test_deep_itm_call(self):
        """Deep ITM call should be close to S - K*exp(-rT)"""
        opt = make_option(S=200, K=50, opt_type="call")
        bs = BlackScholes(opt)
        intrinsic = 200 - 50 * np.exp(-opt.r * opt.T)
        assert abs(bs.price() - intrinsic) / intrinsic < 0.01

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero"""
        bs = BlackScholes(make_option(S=50, K=200, opt_type="call"))
        assert bs.price() < 1.0

    def test_call_increases_with_spot(self):
        """Call price should increase as spot increases"""
        low = BlackScholes(make_option(S=90, opt_type="call")).price()
        high = BlackScholes(make_option(S=110, opt_type="call")).price()
        assert high > low

    def test_put_increases_with_strike(self):
        """Put price should increase as strike increases"""
        low = BlackScholes(make_option(K=100, opt_type="put")).price()
        high = BlackScholes(make_option(K=140, opt_type="put")).price()
        assert high > low

    def test_price_increases_with_volatility(self):
        """Both call and put prices should increase with volatility"""
        call_low = BlackScholes(make_option(vol=0.10, opt_type="call")).price()
        call_high = BlackScholes(make_option(vol=0.30, opt_type="call")).price()
        put_low = BlackScholes(make_option(vol=0.10, opt_type="put")).price()
        put_high = BlackScholes(make_option(vol=0.30, opt_type="put")).price()
        assert call_high > call_low
        assert put_high > put_low

    def test_unsupported_instrument_raises(self):
        """BlackScholes should reject unsupported instruments"""
        from instruments.base import Instrument
        class FakeInstrument(Instrument):
            pass
        with pytest.raises(TypeError, match="BlackScholes cannot price"):
            BlackScholes(FakeInstrument())


class TestGreeks:
    def setup_method(self):
        self.call_bs = BlackScholes(make_option(opt_type="call"))
        self.put_bs = BlackScholes(make_option(opt_type="put"))

    def test_call_delta_bounds(self):
        """Call delta must be in [0, 1]"""
        assert 0 <= self.call_bs.delta <= 1

    def test_put_delta_bounds(self):
        """Put delta must be in [-1, 0]"""
        assert -1 <= self.put_bs.delta <= 0

    def test_put_call_delta_relationship(self):
        """delta_call - delta_put = 1"""
        diff = self.call_bs.delta - self.put_bs.delta
        assert abs(diff - 1.0) < 1e-10

    def test_gamma_positive(self):
        assert self.call_bs.gamma > 0

    def test_gamma_same_for_call_and_put(self):
        assert abs(self.call_bs.gamma - self.put_bs.gamma) < 1e-10

    def test_vega_positive(self):
        assert self.call_bs.vega > 0

    def test_vega_same_for_call_and_put(self):
        assert abs(self.call_bs.vega - self.put_bs.vega) < 1e-10

    def test_call_rho_positive(self):
        assert self.call_bs.rho > 0

    def test_put_rho_negative(self):
        assert self.put_bs.rho < 0

    def test_delta_finite_difference(self):
        h = 0.01
        opt = make_option(opt_type="call")
        bs_up = BlackScholes(make_option(S=opt.S + h, opt_type="call"))
        bs_dn = BlackScholes(make_option(S=opt.S - h, opt_type="call"))
        fd_delta = (bs_up.price() - bs_dn.price()) / (2 * h)
        assert abs(self.call_bs.delta - fd_delta) < 1e-4

    def test_gamma_finite_difference(self):
        h = 0.01
        opt = make_option(opt_type="call")
        bs_up = BlackScholes(make_option(S=opt.S + h, opt_type="call"))
        bs_mid = BlackScholes(make_option(opt_type="call"))
        bs_dn = BlackScholes(make_option(S=opt.S - h, opt_type="call"))
        fd_gamma = (bs_up.price() - 2 * bs_mid.price() + bs_dn.price()) / (h ** 2)
        assert abs(self.call_bs.gamma - fd_gamma) < 1e-4

    def test_vega_finite_difference(self):
        h = 0.0001
        opt = make_option(opt_type="call")
        bs_up = BlackScholes(make_option(vol=opt.vol + h, opt_type="call"))
        bs_dn = BlackScholes(make_option(vol=opt.vol - h, opt_type="call"))
        fd_vega = (bs_up.price() - bs_dn.price()) / (2 * h)
        assert abs(self.call_bs.vega - fd_vega) < 1e-2

    def test_theta_finite_difference(self):
        h = 1/365
        opt = make_option(opt_type="call")
        bs_up = BlackScholes(make_option(T=opt.T + h, opt_type="call"))
        bs_dn = BlackScholes(make_option(T=opt.T - h, opt_type="call"))
        fd_theta = -(bs_up.price() - bs_dn.price()) / (2 * h)
        assert abs(self.call_bs.theta - fd_theta) < 0.1

    def test_rho_finite_difference(self):
        h = 0.0001
        opt = make_option(opt_type="call")
        bs_up = BlackScholes(make_option(r=opt.r + h, opt_type="call"))
        bs_dn = BlackScholes(make_option(r=opt.r - h, opt_type="call"))
        fd_rho = (bs_up.price() - bs_dn.price()) / (2 * h)
        assert abs(self.call_bs.rho - fd_rho) < 0.1

    def test_atm_call_delta_near_half(self):
        atm = BlackScholes(make_option(S=100, K=100, opt_type="call"))
        assert 0.45 < atm.delta < 0.75

    def test_deep_itm_call_delta_near_one(self):
        deep_itm = BlackScholes(make_option(S=200, K=50, opt_type="call"))
        assert deep_itm.delta > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        deep_otm = BlackScholes(make_option(S=50, K=200, opt_type="call"))
        assert deep_otm.delta < 0.01


class TestMonteCarlo:
    def setup_method(self):
        np.random.seed(42)

    def test_call_price_positive(self):
        mc = MonteCarlo(make_option(opt_type="call"))
        assert mc.price() > 0

    def test_put_price_positive(self):
        mc = MonteCarlo(make_option(opt_type="put"))
        assert mc.price() > 0

    def test_call_upper_bound(self):
        opt = make_option(opt_type="call")
        assert MonteCarlo(opt).price() <= opt.S

    def test_put_upper_bound(self):
        opt = make_option(opt_type="put")
        assert MonteCarlo(opt).price() <= opt.K * np.exp(-opt.r * opt.T)

    def test_deep_itm_call(self):
        opt = make_option(S=200, K=50, opt_type="call")
        mc = MonteCarlo(opt, n_paths=50_000)
        intrinsic = 200 - 50 * np.exp(-opt.r * opt.T)
        assert abs(mc.price() - intrinsic) / intrinsic < 0.02

    def test_deep_otm_call(self):
        mc = MonteCarlo(make_option(S=50, K=200, opt_type="call"))
        assert mc.price() < 1.0

    def test_call_increases_with_spot(self):
        low = MonteCarlo(make_option(S=90, opt_type="call"), n_paths=50_000).price()
        high = MonteCarlo(make_option(S=110, opt_type="call"), n_paths=50_000).price()
        assert high > low

    def test_put_increases_with_strike(self):
        low = MonteCarlo(make_option(K=100, opt_type="put"), n_paths=50_000).price()
        high = MonteCarlo(make_option(K=140, opt_type="put"), n_paths=50_000).price()
        assert high > low

    def test_price_increases_with_volatility(self):
        low = MonteCarlo(make_option(vol=0.10, opt_type="call"), n_paths=50_000).price()
        high = MonteCarlo(make_option(vol=0.30, opt_type="call"), n_paths=50_000).price()
        assert high > low

    def test_put_call_parity(self):
        """MC call - MC put should approximate S - K*exp(-rT)"""
        opt_c = make_option(opt_type="call")
        opt_p = make_option(opt_type="put")
        mc_c = MonteCarlo(opt_c, n_paths=100_000).price()
        mc_p = MonteCarlo(opt_p, n_paths=100_000).price()
        lhs = mc_c - mc_p
        rhs = opt_c.S - opt_c.K * np.exp(-opt_c.r * opt_c.T)
        assert abs(lhs - rhs) < 1.0

    def test_mc_call_converges_to_bs(self):
        opt = make_option(opt_type="call")
        bs_price = BlackScholes(opt).price()
        mc_price = MonteCarlo(opt, n_paths=100_000).price()
        assert abs(mc_price - bs_price) / bs_price < 0.05

    def test_mc_put_converges_to_bs(self):
        opt = make_option(opt_type="put")
        bs_price = BlackScholes(opt).price()
        mc_price = MonteCarlo(opt, n_paths=100_000).price()
        assert abs(mc_price - bs_price) / bs_price < 0.05

    def test_mc_atm_call_converges_to_bs(self):
        opt = make_option(S=100.0, K=100.0, opt_type="call")
        bs_price = BlackScholes(opt).price()
        mc_price = MonteCarlo(opt, n_paths=100_000).price()
        assert abs(mc_price - bs_price) / bs_price < 0.05

    def test_mc_itm_put_converges_to_bs(self):
        opt = make_option(S=90.0, K=110.0, opt_type="put")
        bs_price = BlackScholes(opt).price()
        mc_price = MonteCarlo(opt, n_paths=100_000).price()
        assert abs(mc_price - bs_price) / bs_price < 0.05

    def test_mc_accuracy_improves_with_paths(self):
        opt = make_option(opt_type="call")
        bs_price = BlackScholes(opt).price()
        err_few = abs(MonteCarlo(opt, n_paths=1_000).price() - bs_price)
        err_many = abs(MonteCarlo(opt, n_paths=200_000).price() - bs_price)
        assert err_many < err_few

    def test_unsupported_instrument_raises(self):
        from instruments.base import Instrument
        class FakeInstrument(Instrument):
            pass
        with pytest.raises(TypeError, match="MonteCarlo cannot price"):
            MonteCarlo(FakeInstrument())


class TestDCF:
    def test_zcb_price(self):
        """ZCB price = FV * exp(-r * T)"""
        zcb = ZeroCouponBond(fv=1000, mat=2.0)
        dcf = DCF(zcb, r=0.05)
        expected = 1000 * np.exp(-0.05 * 2.0)
        assert dcf.price() == pytest.approx(expected)

    def test_zcb_price_zero_rate(self):
        zcb = ZeroCouponBond(fv=1000, mat=5.0)
        dcf = DCF(zcb, r=0.0)
        assert dcf.price() == pytest.approx(1000.0)

    def test_zcb_price_decreases_with_rate(self):
        zcb = ZeroCouponBond(fv=1000, mat=5.0)
        low = DCF(zcb, r=0.02).price()
        high = DCF(zcb, r=0.10).price()
        assert low > high

    def test_zcb_price_decreases_with_maturity(self):
        short = DCF(ZeroCouponBond(fv=1000, mat=1.0), r=0.05).price()
        long = DCF(ZeroCouponBond(fv=1000, mat=10.0), r=0.05).price()
        assert short > long

    def test_cb_price_greater_than_zcb(self):
        """Coupon bond should be worth more than ZCB with same FV/mat/rate"""
        r = 0.05
        zcb = ZeroCouponBond(fv=1000, mat=5.0)
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert DCF(cb, r=r).price() > DCF(zcb, r=r).price()

    def test_cb_price_at_par(self):
        """When coupon rate equals discount rate, bond should price near par"""
        cb = CouponBond(fv=1000, mat=10.0, coupon_rate=0.05, freq=1)
        price = DCF(cb, r=0.05).price()
        assert price == pytest.approx(1000.0, rel=0.01)

    def test_cb_premium_when_coupon_above_rate(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.08, freq=2)
        assert DCF(cb, r=0.04).price() > 1000

    def test_cb_discount_when_coupon_below_rate(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.02, freq=2)
        assert DCF(cb, r=0.06).price() < 1000

    def test_cb_price_positive(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert DCF(cb, r=0.05).price() > 0

    def test_zcb_duration_equals_maturity(self):
        zcb = ZeroCouponBond(fv=1000, mat=7.0)
        assert DCF(zcb, r=0.05).duration() == 7.0

    def test_dv01_positive(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert DCF(cb, r=0.05).dv0x() > 0

    def test_dv01_large_bp_shift_normalized(self):
        """When bp_shift >= 0.01, it should be divided by 100"""
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        dv01_normal = DCF(cb, r=0.05).dv0x(bp_shift=0.0001)
        dv01_large = DCF(cb, r=0.05).dv0x(bp_shift=0.01)
        assert dv01_normal == pytest.approx(dv01_large)

    def test_dv01_increases_with_maturity(self):
        short = DCF(CouponBond(fv=1000, mat=2.0, coupon_rate=0.05, freq=2), r=0.05).dv0x()
        long = DCF(CouponBond(fv=1000, mat=10.0, coupon_rate=0.05, freq=2), r=0.05).dv0x()
        assert long > short

    def test_unsupported_instrument_raises(self):
        from instruments.base import Instrument
        class FakeInstrument(Instrument):
            pass
        with pytest.raises(TypeError, match="DCF cannot price"):
            DCF(FakeInstrument(), r=0.05)
