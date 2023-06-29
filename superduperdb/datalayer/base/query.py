import superduperdb as s
import typing as t
from abc import ABC, abstractmethod


class SelectOne(s.JSONable, ABC):
    @abstractmethod
    def __call__(self, db):
        pass


class Like(s.JSONable, ABC):
    @abstractmethod
    def __call__(self, db):
        pass


class Select(s.JSONable, ABC):
    """
    Abstract base class, encapsulating Select database queries/ datalayer reads.
    This allows the concrete implementation of each datalayer to differ substantially on
    stored properties necessary for querying the DB.
    """

    @abstractmethod
    def is_trivial(self) -> bool:
        # Determines when a select statement is "just" select everything.
        # For example, in SQL: "FROM my_table SELECT *"
        # For example, in MongoDB: "collection.find()"
        pass

    @property
    @abstractmethod
    def select_ids(self) -> 'Select':
        # Converts the query into a query which only returns the id
        # of each column/ document.
        pass

    @abstractmethod
    def select_using_ids(
        self,
        ids: t.List[str],
    ) -> t.Any:
        # Converts the query into a query which sub-selects from the ids specified.
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        # Converts the query into a query which sub-selects based on the specified
        # tag "fold"
        pass

    @abstractmethod
    def __call__(self, db):
        pass


class Insert(s.JSONable, ABC):
    # must implement attribute/ property self.documents
    refresh: bool = True
    verbose: bool = True
    documents: t.List

    @property
    @abstractmethod
    def table(self):
        # extracts the table collection from the object
        pass

    @property
    @abstractmethod
    def select_table(self) -> Select:
        # returns a Select object which selects the table into which the insert
        # was inserted
        pass

    @abstractmethod
    def __call__(self, db):
        pass


class Delete(s.JSONable, ABC):
    @abstractmethod
    def __call__(self, db):
        pass


class Update(s.JSONable, ABC):
    refresh: bool = True
    verbose: bool = True

    @property
    @abstractmethod
    def select(self):
        # converts the update object to a Select object, which selects where
        # the update was made
        pass

    @property
    @abstractmethod
    def select_ids(self):
        # converts the update object to a Select object, which selects where
        # the update was made, and returns only ids
        pass

    @abstractmethod
    def __call__(self, db):
        pass
