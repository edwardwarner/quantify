from abc import ABC, abstractmethod
from instruments import Instrument

class Pricer(ABC):
    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument

    @abstractmethod
    def price(self) -> float:
        ...
