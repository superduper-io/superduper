from abc import ABC, abstractmethod
import typing as t

from bson import ObjectId

from superduperdb.core.model import Model
from superduperdb.core.documents import Document
from superduperdb.datalayer.base.cursor import SuperDuperCursor
from superduperdb.datalayer.base.query import Insert, Select, Update, Delete

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.database import (
        UpdateResult,
        DeleteResult,
        InsertResult,
    )


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
    def insert(self, insert: Insert) -> 'InsertResult':
        pass

    @abstractmethod
    def download_update(
        self, table: str, id: ObjectId, key: str, bytes: bytes
    ) -> Update:
        pass

    def get_cursor(
        self,
        select: Select,
        types: t.Dict,
        features: t.Union[t.Mapping[str, str], None] = None,
        scores: t.Optional[t.List[float]] = None,
    ) -> SuperDuperCursor:
        return SuperDuperCursor(
            self.get_raw_cursor(select),
            id_field=self.id_field,
            types=types,
            features=features if features else select.features,
            scores=scores,
        )

    @abstractmethod
    def get_ids_from_select(self, select: Select) -> t.List[int]:
        pass

    @abstractmethod
    def get_output_from_document(
        self, r: Document, key: str, model: str
    ) -> t.Tuple[t.Dict, t.Any]:
        pass

    @abstractmethod
    def get_raw_cursor(self, select: Select) -> t.Any:
        pass

    @abstractmethod
    def set_content_bytes(self, r: t.Dict, key: str, bytes_: bytes) -> t.Dict:
        pass

    @abstractmethod
    def write_outputs(
        self,
        watcher_info: t.Dict[str, t.Any],
        outputs: t.List[t.Dict],
        _ids: t.List[ObjectId],
    ) -> None:
        pass

    @abstractmethod
    def update(self, update: Update) -> 'UpdateResult':
        pass

    @abstractmethod
    def delete(self, delete: Delete) -> 'DeleteResult':
        pass

    @abstractmethod
    def unset_outputs(self, info: t.Dict[str, t.Any]) -> 'UpdateResult':
        pass
