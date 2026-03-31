from dataclasses import dataclass
from instruments.base import Instrument

from typing import Literal

@dataclass
class EuropeanOption(Instrument):
    S: float
    K: float
    r: float
    vol: float
    T: float
    opt_type: Literal["call", "put"] = "call"
