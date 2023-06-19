from abc import ABC, abstractmethod
import typing as t

from superduperdb.core import Model
from superduperdb.core.documents import Document
from superduperdb.datalayer.base.cursor import SuperDuperCursor
from superduperdb.datalayer.base.query import Insert, Select, Update, Delete


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
    def insert(self, insert: Insert):
        pass

    @abstractmethod
    def download_update(self, table, id, key, bytes):
        pass

    def get_cursor(
        self,
        select: Select,
        types: t.Dict,
        features=None,
        scores=None,
    ):
        return SuperDuperCursor(
            self.get_raw_cursor(select),
            id_field=self.id_field,
            types=types,
            features=features if features else select.features,
            scores=scores,
        )

    @abstractmethod
    def get_ids_from_select(self, select: Select):
        pass

    @abstractmethod
    def get_output_from_document(self, r: Document, key: str, model: str):
        pass

    @abstractmethod
    def get_raw_cursor(self, select: Select):
        pass

    @abstractmethod
    def get_query_for_validation_set(self, validation_set):
        pass

    @abstractmethod
    def insert_validation_data(self, tmp, identifier):
        pass

    @abstractmethod
    def set_content_bytes(self, r, key, bytes_):
        pass

    @abstractmethod
    def write_outputs(self, info, outputs, _ids):
        pass

    @abstractmethod
    def update(self, update: Update):
        pass

    @abstractmethod
    def delete(self, delete: Delete):
        pass

    @abstractmethod
    def unset_outputs(self, info):
        pass

    @abstractmethod
    def show_validation_sets(self):
        pass
