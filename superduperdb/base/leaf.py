import typing as t
from abc import ABC, abstractmethod, abstractproperty

from superduperdb.base.config import BytesEncoding


class Leaf(ABC):
    @abstractproperty
    def unique_id(self):
        pass

    @abstractmethod
    def encode(
        self,
        bytes_encoding: t.Optional[BytesEncoding] = None,
        leaf_types_to_keep: t.Sequence = (),
    ):
        pass

    @classmethod
    @abstractmethod
    def decode(cls, r, db):
        pass
