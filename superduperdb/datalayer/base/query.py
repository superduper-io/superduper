from abc import ABC, abstractmethod

from typing import List, Any

from superduperdb.misc.serialization import convert_from_types_to_bytes


class Select(ABC):
    """
    Abstract base class, encapsulating Select database queries/ datalayer reads.
    This allows the concrete implementation of each datalayer to differ substantially on
    stored properties necessary for querying the DB.
    """

    @property
    @abstractmethod
    def is_trivial(self) -> bool:
        # Determines when a select statement is "just" select everything.
        # For example, in SQL: "FROM my_table SELECT *"
        # For example, in MongoDB: "collection.find()"
        pass

    @property
    @abstractmethod
    def select_only_id(self) -> 'Select':
        # Converts the query into a query which only returns the id of each column/ document.
        pass

    @abstractmethod
    def select_using_ids(self, ids):
        # Converts the query into a query which sub-selects from the ids specified.
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        # Converts the query into a query which sub-selects based on the specified tag "fold"
        pass


class Insert(ABC):
    # must implement attribute/ property self.documents

    def _to_raw_documents(self, types, type_lookup) -> List[Any]:
        raw_documents = []
        for r in self.documents:
            raw_documents.append(convert_from_types_to_bytes(r, types, type_lookup))
        return raw_documents

    @abstractmethod
    def to_raw(self, types, type_lookup) -> 'Insert':
        # converts the Insert object into an equivalent insert object, but where the component
        # types are dumped as bytes
        pass

    @property
    @abstractmethod
    def table(self):
        # extracts the table name from the object
        pass

    @property
    @abstractmethod
    def select_table(self) -> Select:
        # returns a Select object which selects the table into which the insert was inserted
        pass


class Delete(ABC):
    ...


class Update(ABC):
    @abstractmethod
    def to_raw(self, types, type_lookup):
        # converts the Update object into an equivalent Update object, but where the component
        # types are dumped as bytes in the update
        pass

    @property
    @abstractmethod
    def select(self):
        # converts the update object to a Select object, which selects where the update was made
        pass

    @property
    @abstractmethod
    def select_ids(self):
        # converts the update object to a Select object, which selects where the update was made
        # and returns only ids
        pass
