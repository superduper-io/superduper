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
    def build_metadata(self):
        """
        Build a default metadata store based on current connection.
        """
        pass

    @abstractmethod
    def build_artifact_store(self):
        """
        Build a default artifact store based on current connection.
        """
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the databackend.
        """
        pass
