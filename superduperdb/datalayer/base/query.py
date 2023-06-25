import superduperdb as s
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import Field
from superduperdb.core.documents import Document
from superduperdb.core.suri import URIDocument


class Select(s.JSONable, ABC):
    """
    Abstract base class, encapsulating Select database queries/ datalayer reads.
    This allows the concrete implementation of each datalayer to differ substantially on
    stored properties necessary for querying the DB.
    """

    download: bool = False
    features: t.Optional[t.Dict[str, str]] = None
    filter: t.Optional[t.Dict] = None
    kwargs: t.Dict = Field(default_factory=dict)
    like: t.Optional[URIDocument] = None
    n: int = 100
    one: bool = False
    outputs: t.Optional[URIDocument] = None
    projection: t.Optional[t.Dict[str, int]] = None
    raw: bool = False
    similar_first: bool = False
    vector_index: t.Optional[str] = None

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
        # Converts the query into a query which only returns the id
        # of each column/ document.
        pass

    @abstractmethod
    def select_using_ids(
        self, ids: t.List[str], features: t.Optional[t.Dict[str, str]] = None
    ) -> t.Any:
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
    documents: t.List[Document]

    refresh: bool = True
    verbose: bool = True

    @property
    @abstractmethod
    def table(self):
        # extracts the table name from the object
        pass

    @property
    @abstractmethod
    def select_table(self) -> Select:
        # returns a Select object which selects the table into which the insert
        # was inserted
        pass


class Delete(s.JSONable, ABC):
    ...


class Update(ABC):
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
