from abc import ABC, abstractmethod
import dill
import enum
import hashlib
import io
import pickle
import typing as t


class Serializer(enum.Enum):
    """
    Enumerates the types of Python serializers to use
    """

    dill = 'dill'
    pickle = 'pickle'
    default = dill

    @property
    def impl(self):
        return dill if self.value == 'dill' else pickle


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

    def _serialize(
        self,
        object: t.Any,
        serializer: Serializer = Serializer.default,
        serializer_kwargs: t.Optional[t.Dict] = None,
    ):
        if serializer == Serializer.default:
            # TODO: this was what was there, but is it right?
            serializer_kwargs = serializer_kwargs or {'recurse': True}

        with io.BytesIO() as f:
            serializer.impl.dump(object, f, **serializer_kwargs)
            return f.getvalue()

    @abstractmethod
    def delete_artifact(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        pass

    def create_artifact(
        self,
        object: t.Any,
        serializer: t.Union[Serializer, str] = Serializer.default,
        serializer_kwargs: t.Optional[t.Dict] = None,
    ):
        """
        Save serialized object in the artifact store.

        :param object: Object to serialize
        :param serializer: Serializer to use
        :param serializer_kwargs: Keyword parameters to pass to serializer
        """
        if isinstance(serializer, str):
            serializer = Serializer(serializer)

        bytes = self._serialize(object, serializer, serializer_kwargs)
        return self._save_artifact(bytes), hashlib.sha1(bytes).hexdigest()

    @abstractmethod
    def _save_artifact(self, serialized: bytes):
        pass

    @abstractmethod
    def _load_bytes(self, file_id):
        pass

    def load_artifact(self, file_id: str, serializer: str):
        """
        Load artifact from artifact store, and deserialize.

        :param file_id: Identifier of artifact in the store
        :param serializer: Serializer to use for deserialization
        """
        bytes = self._load_bytes(file_id)
        fp = io.BytesIO(bytes)
        if serializer == 'dill':
            return dill.load(fp)
        elif serializer == 'pickle':
            return pickle.load(fp)
        else:
            raise NotImplementedError
