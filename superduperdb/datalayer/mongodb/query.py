from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Mapping, Any, List

from superduperdb.datalayer.base import query
from superduperdb.misc.serialization import convert_from_types_to_bytes


@dataclass(frozen=True)
class Select(query.Select):
    collection: str
    filter: Optional[Mapping[str, Any]] = None
    projection: Optional[Mapping[str, Any]] = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    one: bool = False

    def add_fold(self, fold: str) -> 'Select':
        return Select(
            collection=self.collection,
            filter={'_fold': fold, **(self.filter or {})},
            projection=self.projection,
            kwargs=self.kwargs,
            one=self.one,
        )

    @cached_property
    def table(self):
        return self.collection

    @cached_property
    def is_trivial(self) -> bool:
        return not self.filter

    @cached_property
    def select_only_id(self) -> 'Select':
        return Select(
            collection=self.collection, filter=self.filter, projection={'_id': 1}
        )

    def select_using_ids(self, ids) -> 'Select':
        return Select(
            collection=self.collection,
            filter={'_id': {'$in': ids}, **(self.filter if self.filter else {})},
            projection=self.projection,
            kwargs=self.kwargs,
        )

    def update(self, to_update: dict) -> 'Update':
        return Update(
            collection=self.collection,
            filter=self.filter or {},
            update=to_update,
        )


@dataclass(frozen=True)
class Update(query.Update):
    collection: str
    filter: Mapping[str, Any]
    update: Optional[Mapping[str, Any]] = None
    one: bool = False
    replacement: Optional[Mapping[str, Any]] = None

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

    def to_raw(self, types, type_lookup):
        if self.replacement is None:
            assert self.update is not None
            return Update(
                collection=self.collection,
                filter=self.filter,
                update=convert_from_types_to_bytes(self.update, types, type_lookup),
            )
        else:
            return Update(
                collection=self.collection,
                filter=self.filter,
                replacement=convert_from_types_to_bytes(
                    self.replacement, types, type_lookup
                ),
            )


@dataclass(frozen=True)
class Delete(query.Delete):
    collection: str
    filter: Mapping[str, Any]
    one: bool = False

    @cached_property
    def table(self):
        return self.collection


@dataclass(frozen=True)
class Insert(query.Insert):
    collection: str
    documents: List[Any]
    ordered: bool = True
    bypass_document_validation: bool = False

    @cached_property
    def table(self):
        return self.collection

    def to_raw(self, types: dict, type_lookup: dict) -> 'Insert':
        return Insert(
            collection=self.collection,
            documents=self._to_raw_documents(types, type_lookup),
            ordered=self.ordered,
            bypass_document_validation=self.bypass_document_validation,
        )

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
