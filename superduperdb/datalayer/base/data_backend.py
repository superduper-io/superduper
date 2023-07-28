import typing as t
from abc import ABC, abstractmethod

from superduperdb.core.document import Document
from superduperdb.core.model import Model
from superduperdb.datalayer.base.query import Select


class BaseDataBackend(ABC):
    models: t.Dict[str, Model]
    select_cls = Select
    id_field = 'id'

    def __init__(self, conn: t.Any, name: str):
        self.conn = conn
        self.name = name

    @property
    def db(self):
        raise NotImplementedError

    @abstractmethod
    def get_output_from_document(self, r: Document, key: str, model: str):
        pass

    @abstractmethod
    def set_content_bytes(self, r, key, bytes_):
        pass

    @abstractmethod
    def unset_outputs(self, info):
        pass
