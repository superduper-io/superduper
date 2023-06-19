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

    This might include models, data artifacts...
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
        pass

    def create_artifact(
        self,
        object: t.Any,
        serializer: t.Union[Serializer, str] = Serializer.default,
        serializer_kwargs: t.Optional[t.Dict] = None,
    ):
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

    def load_artifact(self, file_id: str, serializer: Serializer = Serializer.default):
        if isinstance(serializer, str):
            serializer = Serializer(serializer)

        bytes = self._load_bytes(file_id)
        fp = io.BytesIO(bytes)
        return serializer.impl.load(fp)
