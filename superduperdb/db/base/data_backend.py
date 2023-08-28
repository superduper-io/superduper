import typing as t
from abc import ABC, abstractmethod


class BaseDataBackend(ABC):
    id_field = 'id'

    def __init__(self, conn: t.Any, name: str):
        self.conn = conn
        self.name = name

    @property
    def db(self):
        raise NotImplementedError

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the databackend.
        """
        pass

    # TODO - not the right place for this
    @abstractmethod
    def set_content_bytes(self, r, key, bytes_):
        pass

    # TODO - not the right place for this
    @abstractmethod
    def unset_outputs(self, info):
        pass
