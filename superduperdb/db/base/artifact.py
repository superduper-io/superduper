import hashlib
import typing as t
from abc import ABC, abstractmethod

from superduperdb.misc.serialization import Info, serializers


class ArtifactStore(ABC):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn

    @abstractmethod
    def delete_artifact(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the artifact store.
        """
        pass

    def create_artifact(self, bytes: t.Any):
        """
        Save serialized object in the artifact store.

        :param object: Object to serialize
        :param bytes: Serialized object
        """
        return self._save_artifact(bytes), hashlib.sha1(bytes).hexdigest()

    @abstractmethod
    def _save_artifact(self, serialized: bytes) -> t.Any:
        pass

    @abstractmethod
    def _load_bytes(self, file_id: str) -> bytes:
        pass

    def load_artifact(self, file_id: str, serializer: str, info: Info = None) -> t.Any:
        """
        Load artifact from artifact store, and deserialize.

        :param file_id: Identifier of artifact in the store
        :param serializer: Serializer to use for deserialization
        """
        bytes = self._load_bytes(file_id)
        serializer_function = serializers[serializer]
        return serializer_function.decode(bytes, info)
