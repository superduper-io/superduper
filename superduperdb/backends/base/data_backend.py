import typing as t
from abc import ABC, abstractmethod

from superduperdb.components.model import APIModel, Model


class BaseDataBackend(ABC):
    id_field = 'id'

    def __init__(self, conn: t.Any, name: str):
        self.conn = conn
        self.name = name
        self.in_memory: bool = False

    @property
    def db(self):
        raise NotImplementedError

    @abstractmethod
    def url(self):
        """
        Databackend connection url
        """
        pass

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

    def create_model_table_or_collection(self, model: t.Union[Model, APIModel]):
        pass

    @abstractmethod
    def get_table_or_collection(self, identifier):
        pass

    def set_content_bytes(self, r, key, bytes_):
        raise NotImplementedError

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the databackend.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Disconnect the client
        """
