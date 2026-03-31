import pytest
from dataclasses import FrozenInstanceError

from instruments.base import Instrument
from instruments.option import EuropeanOption
from instruments.bond import ZeroCouponBond, CouponBond


class TestInstrumentBase:
    def test_is_abstract_base_class(self):
        """Instrument should be an ABC"""
        import abc
        assert issubclass(Instrument, abc.ABC)

    def test_european_option_is_instrument(self):
        opt = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0)
        assert isinstance(opt, Instrument)


class TestEuropeanOption:
    def test_default_opt_type_is_call(self):
        opt = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0)
        assert opt.opt_type == "call"

    def test_explicit_put(self):
        opt = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0, opt_type="put")
        assert opt.opt_type == "put"

    def test_fields_stored(self):
        opt = EuropeanOption(S=101.5, K=110.0, r=0.03, vol=0.25, T=0.5, opt_type="put")
        assert opt.S == 101.5
        assert opt.K == 110.0
        assert opt.r == 0.03
        assert opt.vol == 0.25
        assert opt.T == 0.5
        assert opt.opt_type == "put"

    def test_equality(self):
        a = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0)
        b = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0)
        assert a == b

    def test_inequality_different_strike(self):
        a = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0)
        b = EuropeanOption(S=100, K=110, r=0.05, vol=0.2, T=1.0)
        assert a != b

    def test_inequality_different_opt_type(self):
        a = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0, opt_type="call")
        b = EuropeanOption(S=100, K=100, r=0.05, vol=0.2, T=1.0, opt_type="put")
        assert a != b


class TestZeroCouponBond:
    def test_is_instrument(self):
        zcb = ZeroCouponBond(fv=1000, mat=2.0)
        assert isinstance(zcb, Instrument)

    def test_fields_stored(self):
        zcb = ZeroCouponBond(fv=1000, mat=5.0)
        assert zcb.fv == 1000
        assert zcb.mat == 5.0

    def test_equality(self):
        a = ZeroCouponBond(fv=1000, mat=2.0)
        b = ZeroCouponBond(fv=1000, mat=2.0)
        assert a == b

    def test_inequality(self):
        a = ZeroCouponBond(fv=1000, mat=2.0)
        b = ZeroCouponBond(fv=1000, mat=3.0)
        assert a != b


class TestCouponBond:
    def test_is_instrument(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert isinstance(cb, Instrument)

    def test_fields_stored(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert cb.fv == 1000
        assert cb.mat == 5.0
        assert cb.coupon_rate == 0.06
        assert cb.freq == 2

    def test_derived_fields(self):
        cb = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert cb.adj_rate == pytest.approx(0.03)
        assert cb.coupon == pytest.approx(30.0)
        assert cb.periods == 10
        assert len(cb.cf_times) == 10

    def test_cf_times(self):
        cb = CouponBond(fv=1000, mat=2.0, coupon_rate=0.04, freq=4)
        expected = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert cb.cf_times == pytest.approx(expected)

    def test_equality(self):
        a = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        b = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        assert a == b

    def test_inequality_different_coupon(self):
        a = CouponBond(fv=1000, mat=5.0, coupon_rate=0.06, freq=2)
        b = CouponBond(fv=1000, mat=5.0, coupon_rate=0.08, freq=2)
        assert a != b
