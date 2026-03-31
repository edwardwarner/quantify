import pytest
from dataclasses import FrozenInstanceError

from instruments.base import Instrument
from instruments.option import EuropeanOption


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
