from dataclasses import dataclass
from functools import cached_property

from typing import List, Any

from superduperdb.misc.serialization import convert_from_types_to_bytes


@dataclass(frozen=True)
class Query:
    ...


@dataclass(frozen=True)
class Select(Query):

    @cached_property
    def is_trivial(self) -> bool:
        raise NotImplementedError

    @cached_property
    def select_only_id(self) -> 'Select':
        raise NotImplementedError

    def select_using_ids(self, ids):
        raise NotImplementedError

    def add_fold(self, fold: str) -> 'Select':
        raise NotImplementedError


@dataclass(frozen=True)
class Insert(Query):
    documents: List[Any]

    def _to_raw_documents(self, types, type_lookup):
        raw_documents = []
        for r in self.documents:
            raw_documents.append(convert_from_types_to_bytes(r, types, type_lookup))
        return raw_documents

    def to_raw(self, types, type_lookup):
        raise NotImplementedError

    @cached_property
    def table(self):
        raise NotImplementedError

    @cached_property
    def select_table(self) -> Select:
        raise NotImplementedError


@dataclass(frozen=True)
class Delete(Query):
    ...


@dataclass(frozen=True)
class Update(Query):

    def to_raw(self, types, type_lookup):
        raise NotImplementedError

    @cached_property
    def select(self):
        raise NotImplementedError

    @cached_property
    def select_ids(self):
        raise NotImplementedError
