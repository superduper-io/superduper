from abc import ABC, abstractmethod
import dill
import io
import pickle
from typing import Any, Optional, Dict


class ArtifactStore(ABC):
    """
    Abstraction for storing models, data artifacts, etc. separately from primary data
    """

    def _serialize(
        self, object: Any, serializer: str, serializer_kwargs: Optional[Dict] = None
    ):
        serializer_kwargs = serializer_kwargs or {}
        if serializer == 'pickle':
            with io.BytesIO() as f:
                pickle.dump(object, f, **serializer_kwargs)
                bytes_ = f.getvalue()
        elif serializer == 'dill':
            if not serializer_kwargs:
                serializer_kwargs['recurse'] = True
            with io.BytesIO() as f:
                dill.dump(object, f, **serializer_kwargs)
                bytes_ = f.getvalue()
        else:
            raise NotImplementedError
        return bytes_

    @abstractmethod
    def delete_artifact(self, file_id: str):
        pass

    def create_artifact(
        self,
        object: Any,
        serializer: str,
        serializer_kwargs: Optional[Dict] = None,
    ):
        bytes = self._serialize(object, serializer, serializer_kwargs)
        return self._save_artifact(bytes)
        pass

    @abstractmethod
    def _save_artifact(self, serialized: bytes):
        pass

    @abstractmethod
    def _load_bytes(self, file_id):
        pass

    def load_artifact(self, file_id: str, serializer: str):
        bytes = self._load_bytes(file_id)
        f = io.BytesIO(bytes)
        if serializer == 'pickle':
            return pickle.load(f)
        elif serializer == 'dill':
            return dill.load(f)
        else:
            raise NotImplementedError(f'{serializer} serializer not implemented')
