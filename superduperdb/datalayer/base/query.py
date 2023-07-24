import dataclasses as dc
from superduperdb.core.serializable import Serializable

import typing as t
from abc import ABC, abstractmethod


@dc.dataclass
class SelectOne(ABC, Serializable):
    """
    Base class for queries which return a single line/ record of data
    """

    @abstractmethod
    def __call__(self, db):
        pass


@dc.dataclass
class Like(ABC, Serializable):
    """
    Base class for queries which invoke vector-search
    """

    @abstractmethod
    def __call__(self, db):
        pass


@dc.dataclass
class Select(ABC, Serializable):
    """
    Abstract base class, encapsulating Select database queries/ datalayer reads.
    This allows the concrete implementation of each datalayer to differ substantially on
    stored properties necessary for Serializableing the DB.
    """

    @property
    @abstractmethod
    def select_table(self):
        pass

    @abstractmethod
    def is_trivial(self) -> bool:
        # Determines when a select statement is "just" select everything.
        # For example, in SQL: "FROM my_table SELECT *"
        # For example, in MongoDB: "collection.find()"
        pass

    @property
    @abstractmethod
    def select_ids(self) -> 'Select':
        # Converts the Serializable into a Serializable which only returns the id
        # of each column/ document.
        pass

    @abstractmethod
    def select_using_ids(self, ids: t.Sequence[str]) -> t.Any:
        """
        Create a select using the same Serializable, subset to the specified ids

        :param ids: string ids to which subsetting should occur
        """
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        """
        Create a select which selects the same data, but additionally restricts to the
        fold specified

        :param fold: possible values {'train', 'valid'}
        """
        pass

    @abstractmethod
    def model_update(self, db, model, key, outputs, ids):
        """
        Add outputs of ``model`` to the datalayer ``db``.

        :param db: datalayer
        :param model: model identifier to be updated against
        :param key: key on which model was applied
        :param outputs: (encoded) outputs to be added
        :param ids: ids of input documents corresponding to each output
        """
        pass

    @abstractmethod
    def __call__(self, db):
        """
        Apply Serializable to datalayer

        :param db: datalayer instance
        """
        pass


@dc.dataclass
class Insert(ABC, Serializable):
    """
    Base class for database inserts.

    :param refresh: toggle to ``False`` to suppress job triggering
                    (model computations on new docs)
    :param verbose: toggle tp ``False`` to suppress/reduce stdout
    :param documents: list of documents to insert

    """

    # must implement attribute/ property self.documents
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
        """
        Apply Serializable to datalayer

        :param db: datalayer instance
        """
        pass


@dc.dataclass
class Delete(ABC, Serializable):
    """
    Base class for deleting documents from datalayer
    """

    @abstractmethod
    def __call__(self, db):
        """
        Apply Serializable to datalayer

        :param db: datalayer instance
        """
        pass


@dc.dataclass
class Update(ABC, Serializable):
    """
    Base class for database updates.

    :param refresh: toggle to ``False`` to suppress job triggering
                    (model computations on new docs)
    :param verbose: toggle tp ``False`` to suppress/reduce stdout

    """

    @property
    @abstractmethod
    def select_table(self):
        pass

    @property
    @abstractmethod
    def select(self):
        """
        Converts the update object to a Select object, which selects where
        the update was made.
        """
        pass

    @property
    @abstractmethod
    def select_ids(self):
        """
        Converts the update object to a Select object, which selects where
        the update was made, and returns only ids.
        """
        pass

    @abstractmethod
    def __call__(self, db):
        """
        Apply Serializable to datalayer.

        :param db: datalayer instance
        """
        pass
