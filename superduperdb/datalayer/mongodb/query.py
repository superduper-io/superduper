from dataclasses import dataclass
from functools import cached_property
from bson import ObjectId
from pydantic import Field
from superduperdb.core.documents import Document
from superduperdb.core.suri import URIDocument
from superduperdb.datalayer.base import query
import superduperdb as s
import typing as t

JSON = t.Union[None, bool, float, int, t.Dict, t.List, str]


class Select(s.JSONable, query.Select):
    collection: str
    download: bool = False
    features: t.Optional[t.Dict[str, str]] = None
    filter: t.Optional[t.Dict] = None
    kwargs: t.Dict = Field(default_factory=dict)
    like: t.Optional[URIDocument] = None
    n: int = 100
    one: bool = False
    outputs: t.Optional[URIDocument] = None
    projection: t.Optional[t.Dict[str, int]] = None
    similar_first: bool = False
    vector_index: t.Optional[str] = None

    def add_fold(self, fold: str) -> 'Select':
        return Select(
            collection=self.collection,
            filter={'_fold': fold, **(self.filter or {})},
            projection=self.projection,
            kwargs=self.kwargs,
            one=self.one,
        )

    @cached_property
    def table(self) -> str:
        return self.collection

    @cached_property
    def is_trivial(self) -> bool:
        return not self.filter

    @cached_property
    def select_only_id(self) -> 'Select':
        variables = self.dict()
        variables['projection'] = {'_id': 1}
        return Select(**variables)

    def select_using_ids(
        self, ids: t.List[str], features: t.Optional[t.Dict[str, str]] = None
    ) -> 'Select':
        # NOTE: here we assume that the _id field is ObjectId, although it may not be
        # the case.
        variables = self.dict()
        variables['filter'] = {
            '_id': {'$in': [ObjectId(id_) for id_ in ids]},
            **(self.filter if self.filter else {}),
        }
        if features is not None:
            variables['features'] = features
        return Select(**variables)

    def update(self, to_update: Document) -> 'Update':
        return Update(
            collection=self.collection,
            filter=self.filter or {},
            update=to_update,
        )


@dataclass(frozen=True)
class Update(query.Update):
    collection: str
    filter: t.Dict[str, t.Any]
    update: t.Optional[Document] = None
    one: bool = False
    replacement: t.Optional[Document] = None

    @cached_property
    def table(self):
        return self.collection

    @cached_property
    def select_ids(self) -> Select:
        return Select(
            collection=self.collection,
            filter=self.filter,
            projection={'_id': 1},
        )

    @cached_property
    def select(self) -> Select:
        return Select(
            collection=self.collection,
            filter=self.filter,
            projection={'_id': 1},
        )


@dataclass(frozen=True)
class Delete(query.Delete):
    collection: str
    filter: t.Dict[str, t.Any]
    one: bool = False

    @cached_property
    def table(self):
        return self.collection


@dataclass(frozen=False)
class Insert(query.Insert):
    collection: str
    ordered: bool = True
    bypass_document_validation: bool = False

    @cached_property
    def table(self):
        return self.collection

    @cached_property
    def select_table(self) -> Select:
        return Select(collection=self.collection, filter={})


def set_one_key_in_document(table, id, key, value):
    return Update(
        collection=table,
        filter={'_id': id},
        update={'$set': {key: value}},
        one=True,
    )
