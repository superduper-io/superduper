from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import List, Optional, Mapping

from superduperdb.core.documents import Document


class Select(ABC):
    """
    Abstract base class, encapsulating Select database queries/ datalayer reads.
    This allows the concrete implementation of each datalayer to differ substantially on
    stored properties necessary for querying the DB.
    """

    @property
    @abstractmethod
    def is_trivial(self) -> bool:
        # Determines when a _select statement is "just" _select everything.
        # For example, in SQL: "FROM my_table SELECT *"
        # For example, in MongoDB: "collection.find()"
        pass

    @property
    @abstractmethod
    def select_only_id(self) -> 'Select':
        # Converts the query into a query which only returns the id
        # of each column/ document.
        pass

    @abstractmethod
    def select_using_ids(self, ids, features: Optional[Mapping[str, str]] = None):
        # Converts the query into a query which sub-selects from the ids specified.
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        # Converts the query into a query which sub-selects based on the specified
        # tag "fold"
        pass


@dataclass(frozen=False)
class Insert(ABC):
    # must implement attribute/ property self.documents

    documents: List[Document]

    @property
    @abstractmethod
    def table(self):
        # extracts the table name from the object
        pass

    @property
    @abstractmethod
    def select_table(self) -> Select:
        # returns a Select object which selects the table into which the _insert
        # was inserted
        pass


class Delete(ABC):
    ...


class Update(ABC):
    @property
    @abstractmethod
    def select(self):
        # converts the _update object to a Select object, which selects where
        # the update was made
        pass

    @property
    @abstractmethod
    def select_ids(self):
        # converts the _update object to a Select object, which selects where
        # the update was made, and returns only ids
        pass
