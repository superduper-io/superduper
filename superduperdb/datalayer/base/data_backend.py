from abc import ABC, abstractmethod
import typing as t

from superduperdb.core.model import Model
from superduperdb.core.documents import Document
from superduperdb.datalayer.base.query import Select


class BaseDataBackend(ABC):
    models: t.Dict[str, Model]
    select_cls = Select
    id_field = 'id'

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.conn = conn
        self.name = name

    @abstractmethod
    def get_output_from_document(self, r: Document, key: str, model: str):
        pass

    @abstractmethod
    def set_content_bytes(self, r, key, bytes_):
        pass

    @abstractmethod
    def unset_outputs(self, info):
        pass
