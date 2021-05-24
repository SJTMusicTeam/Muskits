"""
Credit to espnet: https://github.com/espnet/espnet/blob/master/espnet2/iterators/abs_iter_factory.py
"""


from abc import ABC
from abc import abstractmethod
from typing import Iterator


class AbsIterFactory(ABC):
    @abstractmethod
    def build_iter(self, epoch: int, shuffle: bool = None) -> Iterator:
        raise NotImplementedError
