from abc import ABC, abstractmethod

from typing import List, Any

from superduperdb.misc.serialization import convert_from_types_to_bytes


class Select(ABC):

    @property
    @abstractmethod
    def is_trivial(self) -> bool:
        pass

    @property
    @abstractmethod
    def select_only_id(self) -> 'Select':
        pass

    @abstractmethod
    def select_using_ids(self, ids):
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        pass


class Insert(ABC):

    def _to_raw_documents(self, types, type_lookup) -> List[Any]:
        raw_documents = []
        for r in self.documents:
            raw_documents.append(convert_from_types_to_bytes(r, types, type_lookup))
        return raw_documents

    @abstractmethod
    def to_raw(self, types, type_lookup):
        pass

    @property
    @abstractmethod
    def table(self):
        pass

    @property
    @abstractmethod
    def select_table(self) -> Select:
        pass


class Delete(ABC):
    ...


class Update(ABC):

    @abstractmethod
    def to_raw(self, types, type_lookup):
        pass

    @property
    @abstractmethod
    def select(self):
        pass

    @property
    @abstractmethod
    def select_ids(self):
        pass
