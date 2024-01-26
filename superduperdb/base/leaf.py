from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import typing as t

from superduperdb.base.config import BytesEncoding
class Leaf(ABC):
    @abstractproperty
    def unique_id(self):
        pass

    @abstractmethod
    def encode(self, bytes_encoding: t.Optional[BytesEncoding] = None, leaf_types_to_keep: t.Sequence = ()):
        pass

    @abstractclassmethod
    def decode(cls, r, db):
        pass